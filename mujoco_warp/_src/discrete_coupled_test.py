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

"""Discrete integrator: the metric factored with its tendon and actuator couplings."""

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
from mujoco_warp._src import types
from mujoco_warp._src.types import DeterminismType

# A branch (hinge, ball, hinge) with two sibling twigs and two free fruit hung from the twigs by
# stiff, damped spatial tendons: every stem couples a free body to a chain of the branch tree. A
# damped fixed tendon couples the sibling twigs (same tree, outside M's ancestor sparsity) and a
# site actuator with a velocity bias couples the branch tip to a reference site on the other twig.
# `{first}` and `{second}` let a test declare the fruit before the branch (dof order) or after it.
# With `floor`, the fruit collide with a plane that they touch from the first step.
_TREE = """
    <body name="branch" pos="0 0 1">
      <joint name="root" type="hinge" axis="0 1 0" damping=".1"/>
      <geom fromto="0 0 0 .3 0 0" mass=".5"/>
      <body name="mid" pos=".3 0 0">
        <joint name="mid" type="ball" damping=".05"/>
        <geom fromto="0 0 0 .2 0 0" mass=".3"/>
        <body name="twig_a" pos=".2 0 0">
          <joint name="twig_a" type="hinge" axis="0 0 1" damping=".02"/>
          <geom fromto="0 0 0 .1 .05 0" mass=".05"/>
          <site name="spur_a" pos=".1 .05 0"/>
        </body>
        <body name="twig_b" pos=".2 0 0">
          <joint name="twig_b" type="hinge" axis="1 0 0" damping=".02"/>
          <geom fromto="0 0 0 .1 -.05 0" mass=".05"/>
          <site name="spur_b" pos=".1 -.05 0"/>
          <site name="ref_b" pos=".05 -.02 0"/>
        </body>
      </body>
    </body>
"""
_FRUIT = """
    <body name="fruit_a" pos=".6 .05 .9">
      <freejoint/>
      <geom type="sphere" size=".035" mass=".15" contype="{contype}"/>
      <site name="stem_a" pos="0 0 .035"/>
    </body>
    <body name="fruit_b" pos=".6 -.05 .9">
      <freejoint/>
      <geom type="ellipsoid" size=".03 .035 .04" mass=".12" contype="{contype}"/>
      <site name="stem_b" pos=".005 0 .04"/>
    </body>
"""
_FLOOR = '<geom name="floor" type="plane" size="3 3 .1" pos="0 0 .85" contype="0" conaffinity="2"/>'
_XML = """
<mujoco>
  <option integrator="discrete" timestep="{dt}"/>
  <default><geom type="capsule" size=".01" contype="0" conaffinity="0"/></default>
  <worldbody>
    {first}
    {second}
    {floor}
  </worldbody>
  <tendon>
    <spatial stiffness="400" damping="3" springlength="0.06">
      <site site="spur_a"/>
      <site site="stem_a"/>
    </spatial>
    <spatial stiffness="300" damping="2" springlength="0.05">
      <site site="spur_b"/>
      <site site="stem_b"/>
    </spatial>
    <fixed damping=".5">
      <joint joint="twig_a" coef="1"/>
      <joint joint="twig_b" coef="-1"/>
    </fixed>
  </tendon>
  <actuator>
    <general site="spur_a" refsite="ref_b" gear="0 0 1 0 0 0" gainprm="5" biastype="affine" biasprm="0 -5 -.4"/>
    <position joint="root" kp="20" kv="1"/>
  </actuator>
  <keyframe>
    <key qpos="{qpos}" qvel="{qvel}" ctrl=".02 .1"/>
  </keyframe>
</mujoco>
"""
_TREE_QPOS = ".1  .99 .05 .1 .02  .2  -.3"
_TREE_QVEL = ".4  .3 -.2 .5  1.1  -.9"
_FRUIT_QPOS = ".61 .06 .88 1 .1 0 0  .62 -.04 .89 .98 0 .2 0"
_FRUIT_QVEL = ".1 -.2 .3 2 -1 3  -.3 .1 .2 -1 2 1"


def _xml(fruit_first: bool, dt: float = 0.002, floor: bool = False) -> str:
  tree_first = not fruit_first
  fruit = _FRUIT.format(contype=2 if floor else 0)
  return _XML.format(
    dt=dt,
    first=_TREE if tree_first else fruit,
    second=fruit if tree_first else _TREE,
    floor=_FLOOR if floor else "",
    qpos=f"{_TREE_QPOS}  {_FRUIT_QPOS}" if tree_first else f"{_FRUIT_QPOS}  {_TREE_QPOS}",
    qvel=f"{_TREE_QVEL}  {_FRUIT_QVEL}" if tree_first else f"{_FRUIT_QVEL}  {_TREE_QVEL}",
  )


def _metric_rhs(m, d):
  """The right-hand side of the discrete smooth solve, M_hat qacc_smooth = rhs."""
  rhs = wp.zeros_like(d.qacc_smooth)
  wp.launch(
    derivative._eff_rhs,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.enableflags, m.dof_treeid, d.tree_awake, d.qfrc_smooth, d.efm_c, d.efm_ca],
    outputs=[rhs],
  )
  return rhs


def _metric_residual(m, d) -> float:
  """|M_hat qacc_smooth - f| / |f| with the matrix-free metric product."""
  rhs = _metric_rhs(m, d)
  prod = wp.zeros_like(d.qacc_smooth)
  derivative.eff_mul_m(m, d, prod, d.qacc_smooth)
  rhs, prod = rhs.numpy(), prod.numpy()
  return float(np.max(np.linalg.norm(prod - rhs, axis=1) / np.linalg.norm(rhs, axis=1)))


def _matrix_free_metric(m, d) -> np.ndarray:
  """The metric of every world (nworld, nv, nv): the matrix-free product of each unit vector."""
  columns = []
  for j in range(m.nv):
    unit = np.zeros((d.nworld, m.nv), dtype=np.float32)
    unit[:, j] = 1.0
    prod = wp.zeros((d.nworld, m.nv), dtype=float)
    derivative.eff_mul_m(m, d, prod, wp.array(unit, dtype=float))
    columns.append(prod.numpy())
  return np.stack(columns, axis=2)


def _assembled_metric(m, d) -> np.ndarray:
  """The metric of every world (nworld, nv, nv) from the assembled efm_qH elements."""
  rownnz, rowadr, colind = (a.numpy() for a in (m.efm_qH_rownnz, m.efm_qH_rowadr, m.efm_qH_colind))
  efm_qH = d.efm_qH.numpy()
  dense = np.zeros((d.nworld, m.nv, m.nv))
  for i in range(m.nv):
    for adr in range(rowadr[i], rowadr[i] + rownnz[i]):
      dense[:, i, colind[adr]] = dense[:, colind[adr], i] = efm_qH[:, adr]
  return dense


class DiscreteCoupledFactorTest(parameterized.TestCase):
  @parameterized.product(fruit_first=[False, True], nworld=[1, 2])
  def test_qacc_smooth_matches_mujoco_and_pcg(self, fruit_first, nworld):
    """The coupled factor solves the full metric: MuJoCo's PCG solution, without iterating."""
    mjm, mjd, m, d = test_data.fixture(xml=_xml(fruit_first), keyframe=0, nworld=nworld)
    self.assertTrue(m.efm_qH_coupled)
    if nworld == 2:
      qvel = d.qvel.numpy()
      qvel[1] *= -0.5
      d.qvel = wp.array(qvel, dtype=float)

    mjw.forward(m, d)
    self.assertLess(_metric_residual(m, d), 1e-5)
    coupled = d.qacc_smooth.numpy()

    mujoco.mj_forward(mjm, mjd)
    scale = np.max(np.abs(mjd.qacc_smooth))
    np.testing.assert_allclose(coupled[0], mjd.qacc_smooth, rtol=1e-4, atol=1e-5 * scale)

    # the backbone-preconditioned PCG path converges to the same solution
    m.efm_qH_coupled = False
    mjw.forward(m, d)
    np.testing.assert_allclose(coupled, d.qacc_smooth.numpy(), rtol=1e-4, atol=1e-5 * scale)
    if nworld == 2:
      self.assertFalse(np.allclose(coupled[0], coupled[1]))

  @parameterized.parameters(0.001, 0.002, 0.005)
  def test_step_matches_mujoco(self, dt):
    """Ten coupled steps track MuJoCo's discrete integrator."""
    mjm, mjd, m, d = test_data.fixture(xml=_xml(False, dt), keyframe=0)
    self.assertTrue(m.efm_qH_coupled)
    for _ in range(10):
      mujoco.mj_step(mjm, mjd)
      mjw.step(m, d)
    np.testing.assert_allclose(d.qvel.numpy()[0], mjd.qvel, rtol=1e-3, atol=1e-4 * np.max(np.abs(mjd.qvel)))
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, rtol=1e-4, atol=1e-5)

  @parameterized.product(fruit_first=[False, True], nworld=[1, 2])
  def test_assembled_metric_matches_matrix_free(self, fruit_first, nworld):
    """The assembled elements are the matrix the matrix-free metric product applies."""
    mjm, _, m, d = test_data.fixture(xml=_xml(fruit_first), keyframe=0, nworld=nworld)
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, mjm.jnt_qposadr[mjm.joint("root").id]] += 0.3  # another branch pose: other moments
      d.qpos = wp.array(qpos, dtype=float)

    mjw.forward(m, d)
    expected = _matrix_free_metric(m, d)
    np.testing.assert_allclose(_assembled_metric(m, d), expected, rtol=1e-5, atol=1e-6 * np.max(np.abs(expected)))
    if nworld == 2:
      self.assertFalse(np.allclose(expected[0], expected[1]))

  def test_deterministic_atomics(self):
    """Deterministic atomics assemble, factor and solve the same metric, bitwise run to run."""
    _, _, m, d = test_data.fixture(xml=_xml(False), keyframe=0)
    mjw.forward(m, d)
    expected = d.qacc_smooth.numpy()
    m.opt.deterministic = DeterminismType.ATOMICS
    runs = []
    for _ in range(3):
      for field in (d.qacc_smooth, d.efm_qH, d.efm_qHLD):
        field.zero_()
      mjw.forward(m, d)
      runs.append([field.numpy() for field in (d.efm_qH, d.efm_qHLD, d.qacc_smooth)])
    np.testing.assert_allclose(runs[0][2], expected, rtol=1e-5, atol=1e-6 * np.max(np.abs(expected)))
    for run in runs[1:]:
      for first, other in zip(runs[0], run):
        np.testing.assert_array_equal(first, other)

  def test_per_level_factor(self):
    """Per-level launches factor the coupled metric like the fused kernel."""
    _, _, m, d = test_data.fixture(xml=_xml(False), keyframe=0)
    mjw.forward(m, d)
    efm_qHLD, qacc_smooth = d.efm_qHLD.numpy(), d.qacc_smooth.numpy()
    d.efm_qHLD.zero_()
    d.qacc_smooth.zero_()
    with mock.patch.object(smooth, "_factor_sparse_fused", return_value=False):
      mjw.forward(m, d)
    np.testing.assert_allclose(d.efm_qHLD.numpy(), efm_qHLD, rtol=1e-5, atol=1e-6 * np.max(np.abs(efm_qHLD)))
    np.testing.assert_allclose(d.qacc_smooth.numpy(), qacc_smooth, rtol=1e-5, atol=1e-6 * np.max(np.abs(qacc_smooth)))

  def test_layout(self):
    """Rows hold every elimination-tree ancestor; fill follows the dof order."""
    tree_first = mujoco.MjModel.from_xml_string(_xml(False))
    fruit_first = mujoco.MjModel.from_xml_string(_xml(True))
    parent = io.efm_qH_parent(tree_first)
    # each fruit hangs below its twig's dof; the twigs' fixed tendon makes twig_b a child of twig_a
    np.testing.assert_array_equal(parent, [-1, 0, 1, 2, 3, 4, 4, 6, 7, 8, 9, 10, 5, 12, 13, 14, 15, 16])
    layout = io.efm_qH_layout(tree_first, parent)
    self.assertEqual(int(layout["rownnz"].sum()), io.efm_qH_nnz(tree_first))
    # declaring the fruit first moves the branch below both fruit: more fill, same metric
    self.assertGreater(io.efm_qH_nnz(fruit_first), io.efm_qH_nnz(tree_first))

  def test_layout_matches_m_without_couplings(self):
    """With M's own elimination tree and no pairs, the layout and schedules are M's sparse LDL."""
    with (
      mock.patch.multiple(types, M_BLOCK_SCALAR_MAX=0, M_BLOCK_DENSE_MAX=0),
      mock.patch.object(io, "_efm_couplings", return_value=([], [])),
    ):
      mjm, _, m, _ = test_data.fixture(xml=_xml(False), keyframe=0)
      # M's rows hold every ancestor, so the last entry before the diagonal is the parent
      last = mjm.M_rowadr + mjm.M_rownnz - 2
      parent = np.where(mjm.M_rownnz > 1, mjm.M_colind[np.maximum(last, 0)], -1)
      layout = io.efm_qH_layout(mjm, parent)
    self.assertEqual(m.qLD_block_total, 0)
    targets = io._ldl_targets(
      layout["factor_elem"], layout["factor_elem_offsets"], layout["all_updates"], layout["level_offsets"]
    )
    for key, field in (
      ("rownnz", m.M_rownnz),
      ("rowadr", m.M_rowadr),
      ("colind", m.M_colind),
      ("all_updates", m.qLD_all_updates),
      ("level_offsets", m.qLD_level_offsets),
      ("factor_elem", m.qLD_factor_elem),
      ("factor_elem_offsets", m.qLD_factor_elem_offsets),
    ):
      np.testing.assert_array_equal(np.asarray(layout[key]), field.numpy(), err_msg=key)
    for key, field in (
      ("factor_target_elem", m.qLD_factor_target_elem),
      ("factor_target_adr", m.qLD_factor_target_adr),
      ("solve_target_update", m.qLD_solve_target_update),
      ("solve_target_adr", m.qLD_solve_target_adr),
      ("solve_target_offsets", m.qLD_solve_target_offsets),
      ("solve_target_long", m.qLD_solve_target_long),
    ):
      np.testing.assert_array_equal(np.asarray(targets[key]), field.numpy(), err_msg=key)
    self.assertEqual(targets["factor_target_offsets_host"], m.qLD_factor_target_offsets_host)
    self.assertEqual(targets["factor_target_long_host"], m.qLD_factor_target_long_host)

  def test_couplings_compare_enums_by_value(self):
    """Ball and free joint actuators, gain and bias types are read from the integer fields."""
    mjm = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <option integrator="discrete"/>
        <worldbody>
          <body name="ball"><joint name="ball" type="ball"/><geom size=".1" mass="1"/></body>
          <body name="free" pos="1 0 0"><freejoint name="free"/><geom size=".1" mass="1"/></body>
          <body name="slider" pos="2 0 0"><joint name="slide" type="slide"/><geom size=".1" mass="1"/></body>
        </worldbody>
        <actuator>
          <general joint="ball" gear="1 0 0" biastype="affine" biasprm="0 -1 0"/>
          <general joint="free" gear="1 0 0 0 0 0" gaintype="affine" gainprm="1 .5 0"/>
          <motor joint="slide"/>
        </actuator>
      </mujoco>
      """
    )
    self.assertEqual(io._efm_actuator_dofs(mjm, 0), [0, 1, 2])
    self.assertEqual(io._efm_actuator_dofs(mjm, 1), [3, 4, 5, 6, 7, 8])
    tendons, actuators = io._efm_couplings(mjm)
    self.assertEqual(tendons, [])
    self.assertEqual([(a, dofs) for a, dofs in actuators], [(0, [0, 1, 2]), (1, [3, 4, 5, 6, 7, 8])])

  def test_backbone_without_couplings(self):
    """No coupled factor outside discrete, or when no tendon/actuator couples two dofs."""
    implicitfast = mujoco.MjModel.from_xml_string(_xml(False).replace('integrator="discrete"', 'integrator="implicitfast"'))
    self.assertIsNone(io.efm_qH_parent(implicitfast))
    self.assertFalse(mjw.put_model(implicitfast).efm_qH_coupled)
    uncoupled = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <option integrator="discrete"/>
        <worldbody>
          <body><joint name="hinge" type="hinge" damping="1"/><geom size=".1" mass="1"/></body>
        </worldbody>
        <actuator><position joint="hinge" kp="1" kv=".1"/></actuator>
      </mujoco>
      """
    )
    self.assertIsNone(io.efm_qH_parent(uncoupled))
    self.assertFalse(mjw.put_model(uncoupled).efm_qH_coupled)


class DiscreteCoupledRolloutTest(parameterized.TestCase):
  """Discrete rollouts of the branch with its floor contacts, deterministic, bitwise run to run."""

  def _rollout(self, solver: str, nstep: int, graph: bool):
    _, _, m, d = test_data.fixture(
      xml=_xml(False, floor=True),
      keyframe=0,
      nworld=2,
      overrides={"opt.solver": solver},
      nconmax=64,
      njmax=256,
    )
    self.assertTrue(m.efm_qH_coupled)
    m.opt.deterministic = DeterminismType.ALL
    qvel = d.qvel.numpy()
    qvel[1] *= -0.5
    d.qvel = wp.array(qvel, dtype=float)

    nefc = []
    mjw.step(m, d)  # compile outside a capture
    nefc.append(d.nefc.numpy().max())
    if graph:
      with wp.ScopedCapture() as capture:  # recorded, not run
        mjw.step(m, d)
      step = lambda: wp.capture_launch(capture.graph)  # noqa: E731
    else:
      step = lambda: mjw.step(m, d)  # noqa: E731
    for _ in range(nstep - 1):
      step()
      nefc.append(d.nefc.numpy().max())
    self.assertGreater(max(nefc), 0)  # the contacts are active: the solver iterates
    return [a.numpy() for a in (d.qpos, d.qvel, d.qacc, d.qacc_warmstart, d.efm_qHLD)]

  @parameterized.product(solver=["NEWTON", "CG"], graph=[False, True])
  def test_rollout_bitwise(self, solver, graph):
    """Two rollouts, eager or replayed from a captured graph, agree to the bit."""
    if graph and not wp.get_device().is_cuda:
      self.skipTest("CUDA graph capture needs a CUDA device")
    first, second = (self._rollout(solver, 20, graph) for _ in range(2))
    for a, b in zip(first, second):
      np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
  wp.init()
  absltest.main()
