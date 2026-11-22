# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CG preconditioner of CONNECT, WELD and JOINT rows: the discrete metric plus their stiffness."""

import re
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import derivative
from mujoco_warp._src import io
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src.types import DeterminismType


def _collars(
  n: int = 6, weld: bool = False, tendon: bool = False, mimic: bool = False, jacobian: str = "sparse", tolerance: float = 1e-4
) -> str:
  """A chain of free bodies (the collars of a branch) with stiff CONNECT rows between neighbours.

  With `weld`, every third equality is a WELD; with `tendon`, a stiff spatial tendon also couples
  the first and last body (a loop, so the metric's coupled factor holds fill the rows must join);
  with `mimic`, a three-hinge finger on its own tree is held by stiff JOINT rows (hinge 1 mimics
  hinge 0, hinge 2 is locked), like the mimic joints of a gripper.
  """
  rng = np.random.default_rng(0)
  bodies, equalities, qpos, qvel = [], [], [], []
  for i in range(n):
    bodies.append(
      f'<body name="b{i}" pos="{0.2 * i} 0 1"><freejoint/><geom type="sphere" size=".03" mass="{0.2 + 0.05 * i}"/>'
      f'<site name="s{i}" pos=".05 0 0"/><site name="p{i}" pos="-.05 0 0"/></body>'
    )
    qpos.append(f"{0.2 * i} 0 1 1 0 0 0")
    qvel.append(" ".join(f"{v:.3f}" for v in rng.normal(size=6) * 0.2))
    if i:
      kind = "weld" if weld and i % 3 == 0 else "connect"
      equalities.append(f'<{kind} site1="s{i - 1}" site2="p{i}" solref="-4000 -40" solimp=".95 .99 .001 .5 2"/>')
  if mimic:
    links = "".join(
      f'<body name="f{j}" pos="{0.1 if j else -0.5} 0 {0 if j else 1}"><joint name="h{j}" type="hinge" axis="0 1 0"/>'
      f'<geom type="capsule" size=".01" fromto="0 0 0 .1 0 0" mass="{0.05 - 0.01 * j}"/>'
      for j in range(3)
    )
    bodies.append(links + "</body>" * 3)
    equalities.append('<joint joint1="h0" joint2="h1" polycoef="0 -1 0 0 0" solref="-4000 -40"/>')
    equalities.append('<joint joint1="h2" polycoef=".05 0 0 0 0" solref="-4000 -40"/>')
    qpos.append("0.1 0 0")
    qvel.append("0.3 -0.2 0.1")
  tendons = ""
  if tendon:
    spatial = f'<spatial stiffness="300" damping="2" springlength=".5"><site site="s0"/><site site="p{n - 1}"/></spatial>'
    tendons = f"<tendon>{spatial}</tendon>"
  return f"""
  <mujoco>
    <option integrator="discrete" timestep=".002" jacobian="{jacobian}" solver="CG" iterations="200" tolerance="{tolerance}"/>
    <worldbody>{"".join(bodies)}</worldbody>
    <equality>{"".join(equalities)}</equality>
    {tendons}
    <keyframe><key qpos="{"  ".join(qpos)}" qvel="{"  ".join(qvel)}"/></keyframe>
  </mujoco>
  """


def _dense_jacobian(d, worldid: int, nv: int) -> np.ndarray:
  """The sparse constraint Jacobian of a world (nefc, nv)."""
  nefc = int(d.nefc.numpy()[worldid])
  rownnz, rowadr = d.efc.J_rownnz.numpy()[worldid], d.efc.J_rowadr.numpy()[worldid]
  colind, vals = d.efc.J_colind.numpy()[worldid, 0], d.efc.J.numpy()[worldid, 0]
  jac = np.zeros((nefc, nv))
  for r in range(nefc):
    for k in range(rownnz[r]):
      jac[r, colind[rowadr[r] + k]] = vals[rowadr[r] + k]
  return jac


def _dense_metric(m, d, worldid: int) -> np.ndarray:
  """M_hat of a world: the coupled factor's assembled elements, or the backbone qH."""
  dense = np.zeros((m.nv, m.nv))
  if m.efm_qH_coupled:
    rownnz, rowadr, colind = (a.numpy() for a in (m.efm_qH_rownnz, m.efm_qH_rowadr, m.efm_qH_colind))
    values = d.efm_qH.numpy()[worldid]
  else:
    rownnz, rowadr, colind = (a.numpy() for a in (m.M_rownnz, m.M_rowadr, m.M_colind))
    values = d.qH.numpy()[worldid]
  for i in range(m.nv):
    for adr in range(rowadr[i], rowadr[i] + rownnz[i]):
      dense[i, colind[adr]] = dense[colind[adr], i] = values[adr]
  return dense


def _reference(m, d, worldid: int) -> np.ndarray:
  """M_hat + sum D J^T J over the active CONNECT, WELD and JOINT rows of a world, in float64."""
  jac = _dense_jacobian(d, worldid, m.nv)
  ids = d.efc.id.numpy()[worldid]
  dvals = d.efc.D.numpy()[worldid]
  nrow = m.efq_qH_eq_nrow.numpy()
  rows = [r for r in range(int(d.ne.numpy()[worldid])) if nrow[ids[r]] > 0]
  return _dense_metric(m, d, worldid) + (jac[rows].T * dvals[rows]) @ jac[rows]


def _prec(m, d):
  """Builds the preconditioner of the data's current rows; returns the solver context holding it."""
  ctx = solver._create_solver_context(m, d)
  derivative.eff_equality_prec(m, d, ctx.efq_H, ctx.efq_HLD, ctx.efq_diag_inv, ctx.efq_row0)
  return ctx


def _layout_dense(mjm, values: np.ndarray) -> np.ndarray:
  """Elements (nefq_qH,) of the preconditioner layout as a dense symmetric matrix."""
  layout = io.efm_qH_layout(mjm, io.efm_qH_elimination(mjm, equality=True), equality=True)
  dense = np.zeros((mjm.nv, mjm.nv))
  for i in range(mjm.nv):
    for adr in range(layout["rowadr"][i], layout["rowadr"][i] + layout["rownnz"][i]):
      dense[i, layout["colind"][adr]] = dense[layout["colind"][adr], i] = values[adr]
  return dense


class CgEqualityPrecTest(parameterized.TestCase):
  @parameterized.product(weld=[False, True], tendon=[False, True], mimic=[False, True], nworld=[1, 2])
  def test_assembled_matrix(self, weld, tendon, mimic, nworld):
    """Elements are the metric plus D * J^T J of every active row of a covered equality."""
    mjm, _, m, d = test_data.fixture(xml=_collars(weld=weld, tendon=tendon, mimic=mimic), keyframe=0, nworld=nworld)
    self.assertTrue(m.efq_qH_active)
    self.assertEqual(m.efm_qH_coupled, tendon)
    if nworld == 2:
      qvel = d.qvel.numpy()
      qvel[1] *= -0.5
      d.qvel = wp.array(qvel, dtype=float)
      active = d.eq_active.numpy()
      active[1, 1] = False  # an equality that is inactive in the second world only
      d.eq_active = wp.array(active, dtype=bool)
    mjw.forward(m, d)
    ctx = _prec(m, d)
    if nworld == 2:
      ne = d.ne.numpy()
      self.assertEqual(int(ne[0] - ne[1]), int(m.efq_qH_eq_nrow.numpy()[1]))
      self.assertEqual(ctx.efq_row0.numpy()[1, 1], -1)
      self.assertGreaterEqual(ctx.efq_row0.numpy()[0, 1], 0)
    for worldid in range(nworld):
      expected = _reference(m, d, worldid)
      actual = _layout_dense(mjm, ctx.efq_H.numpy()[worldid])
      np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6 * np.max(np.abs(expected)))

  @parameterized.product(tendon=[False, True], mimic=[False, True], nworld=[1, 2])
  def test_solve_inverts_the_stiffened_metric(self, tendon, mimic, nworld):
    """The factor solves P z = g: the dense inverse of the metric plus the rows' stiffness."""
    _, _, m, d = test_data.fixture(xml=_collars(tendon=tendon, mimic=mimic), keyframe=0, nworld=nworld)
    mjw.forward(m, d)
    ctx = _prec(m, d)
    g = np.random.default_rng(1).normal(size=(nworld, m.nv)).astype(np.float32)
    z = wp.zeros((nworld, m.nv), dtype=float)
    derivative.eff_solve_equality_prec(m, d, z, wp.array(g, dtype=float), ctx.efq_HLD, ctx.efq_diag_inv)
    for worldid in range(nworld):
      expected = np.linalg.solve(_reference(m, d, worldid), g[worldid].astype(np.float64))
      np.testing.assert_allclose(z.numpy()[worldid], expected, rtol=1e-3, atol=1e-4 * np.max(np.abs(expected)))

  def test_layout_is_the_symbolic_cholesky_structure(self):
    """Rows hold M's pattern, the couplings and the rows' dofs, plus the fill they cause."""
    mjm = mujoco.MjModel.from_xml_string(_collars(weld=True, tendon=True, mimic=True))
    pattern = np.zeros((mjm.nv, mjm.nv), dtype=bool)
    for i in range(mjm.nv):
      cols = mjm.M_colind[mjm.M_rowadr[i] : mjm.M_rowadr[i] + mjm.M_rownnz[i]]
      pattern[i, cols] = pattern[cols, i] = True
    for group in io._efm_coupling_groups(mjm):
      pattern[np.ix_(group, group)] = True
    groups = io._efm_equality_groups(mjm)
    for _, _, dofs in groups:
      pattern[np.ix_(dofs, dofs)] = True
    for k in reversed(range(mjm.nv)):
      lower = np.flatnonzero(pattern[k, :k])
      pattern[np.ix_(lower, lower)] = True
    elimination = io.efm_qH_elimination(mjm, equality=True)
    for i in range(mjm.nv):
      np.testing.assert_array_equal(elimination[1][i], np.flatnonzero(pattern[i, :i]))
    # the equalities add couplings the tendons and actuators alone do not have
    self.assertGreater(io.efm_qH_nnz(mjm), 0)
    self.assertGreater(sum(len(row) for row in elimination[1]), sum(len(row) for row in io.efm_qH_elimination(mjm)[1]))
    # every (i, j) of every equality's dofs is listed once, diagonal included
    layout = io.efm_qH_layout(mjm, elimination, equality=True)
    self.assertEqual(len(layout["eq_pair"]), sum(len(dofs) * (len(dofs) + 1) // 2 for _, _, dofs in groups))
    self.assertEqual(layout["eq_adr"][-1], len(layout["eq_pair"]))

  def test_cg_converges_in_far_fewer_iterations(self):
    """The factor solves the stiff rows, so fewer iterations remain."""
    iterations, qacc = {}, {}
    for active in (True, False):
      _, _, m, d = test_data.fixture(xml=_collars(n=8), keyframe=0)
      m.efq_qH_active = active
      mjw.forward(m, d)
      iterations[active] = int(d.solver_niter.numpy()[0])
      qacc[active] = d.qacc.numpy()[0]
    self.assertLess(iterations[True], iterations[False] // 3)
    np.testing.assert_allclose(qacc[True], qacc[False], rtol=1e-3, atol=1e-4 * np.max(np.abs(qacc[False])))
    # both agree with MuJoCo's constrained acceleration
    mjm, mjd, _, _ = test_data.fixture(xml=_collars(n=8), keyframe=0)
    mujoco.mj_forward(mjm, mjd)
    np.testing.assert_allclose(qacc[True], mjd.qacc, rtol=2e-2, atol=2e-2 * np.max(np.abs(mjd.qacc)))

  def test_cg_with_mimic_rows(self):
    """The mimic and lock rows of a finger are solved by the factor along with the collars'."""
    iterations, qacc = {}, {}
    for active in (True, False):
      _, _, m, d = test_data.fixture(xml=_collars(n=8, mimic=True), keyframe=0)
      m.efq_qH_active = active
      mjw.forward(m, d)
      iterations[active] = int(d.solver_niter.numpy()[0])
      qacc[active] = d.qacc.numpy()[0]
    self.assertLess(iterations[True], iterations[False] // 3)
    np.testing.assert_allclose(qacc[True], qacc[False], rtol=1e-3, atol=1e-4 * np.max(np.abs(qacc[False])))

  def test_deterministic_atomics(self):
    """Deterministic atomics assemble, factor and solve the same preconditioner bitwise."""
    _, _, m, d = test_data.fixture(xml=_collars(tendon=True), keyframe=0)
    m.opt.deterministic = DeterminismType.ATOMICS
    mjw.forward(m, d)
    runs = []
    for _ in range(3):
      ctx = _prec(m, d)
      runs.append([ctx.efq_H.numpy(), ctx.efq_HLD.numpy(), ctx.efq_diag_inv.numpy()])
    for run in runs[1:]:
      for first, other in zip(runs[0], run):
        np.testing.assert_array_equal(first, other)

  def test_per_level_factor(self):
    """Per-level launches factor the preconditioner like the fused kernel."""
    _, _, m, d = test_data.fixture(xml=_collars(tendon=True), keyframe=0)
    mjw.forward(m, d)
    expected = _prec(m, d).efq_HLD.numpy()
    with mock.patch.object(smooth, "_factor_sparse_fused", return_value=False):
      actual = _prec(m, d).efq_HLD.numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6 * np.max(np.abs(expected)))

  def test_inactive_without_rows_or_with_other_integrators(self):
    """No preconditioner: no CONNECT, WELD or JOINT rows, dense Jacobian, Newton or implicitfast."""
    xml = _collars()
    self.assertTrue(mjw.put_model(mujoco.MjModel.from_xml_string(xml)).efq_qH_active)
    self.assertTrue(mjw.put_model(mujoco.MjModel.from_xml_string(_collars(n=1, mimic=True))).efq_qH_active)
    no_rows = re.sub(r"<equality>.*</equality>", "", xml, flags=re.S)
    self.assertFalse(mjw.put_model(mujoco.MjModel.from_xml_string(no_rows)).efq_qH_active)
    self.assertFalse(mjw.put_model(mujoco.MjModel.from_xml_string(_collars(jacobian="dense"))).efq_qH_active)
    implicitfast = xml.replace('integrator="discrete"', 'integrator="implicitfast"')
    self.assertFalse(mjw.put_model(mujoco.MjModel.from_xml_string(implicitfast)).efq_qH_active)
    # the solver builds nothing for Newton, however the model was put
    newton = mjw.put_model(mujoco.MjModel.from_xml_string(xml.replace('solver="CG"', 'solver="Newton"')))
    self.assertTrue(newton.efq_qH_active)
    self.assertFalse(solver._use_equality_prec(newton))
    _, _, m, d = test_data.fixture(xml=xml, keyframe=0)
    self.assertIsNotNone(solver._create_solver_context(m, d).efq_HLD)
    m.efq_qH_active = False
    self.assertIsNone(solver._create_solver_context(m, d).efq_HLD)


if __name__ == "__main__":
  absltest.main()
