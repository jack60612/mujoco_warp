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

"""Sparse LDL factorization paths against the original per-level kernels; backsolve block widths."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import smooth
from mujoco_warp._src import types
from mujoco_warp._src import warp_util


def _tree_xml() -> str:
  """A free-floating branched tree above M_BLOCK_DENSE_MAX dofs (sparse LDL) plus a small chain.

  Joint armature keeps M well conditioned, so GPU reorderings of float32 atomics stay at rounding
  level.
  """

  def chain(n: int, axis: str, side: int = 0) -> str:
    bodies = []
    for depth in range(n):
      joints = f'<joint type="hinge" axis="{axis}"/><joint type="hinge" axis="0 0 1"/>'
      branch = chain(2, "0 1 0") if side and depth == n // 2 else ""
      bodies.append(f'<body pos="0 .02 .05">{joints}<geom type="capsule" size=".02 .025"/>{branch}')
    return "".join(bodies) + "</body>" * n

  branches = "".join(f'<body pos="{0.1 * b} 0 0">{chain(10, "1 0 0", side=1)}</body>' for b in range(3))
  small = '<body pos="1 0 1"><joint type="hinge" axis="0 1 0"/><geom type="sphere" size=".05"/>'
  small += '<body pos="0 0 .1"><joint type="hinge" axis="1 0 0"/><geom type="sphere" size=".05"/></body></body>'
  return (
    '<mujoco><default><joint armature=".01"/></default>'
    '<worldbody><body pos="0 0 2"><freejoint/><geom type="box" size=".1 .1 .1"/>'
    f"{branches}</body>{small}</worldbody></mujoco>"
  )


@wp.kernel
def _qLD_acc_reference(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # In:
  qLD_updates_: wp.array[wp.vec3i],
  L_in: wp.array2d[float],
  # Out:
  L_out: wp.array2d[float],
):
  """The per-level factorization kernel the fused kernel replaced (one launch per level)."""
  worldid, nodeid = wp.tid()
  update = qLD_updates_[nodeid]
  i, k, Madr_ki = update[0], update[1], update[2]
  Madr_i = M_rowadr[i]
  diag_k = M_rowadr[k] + M_rownnz[k] - 1
  tmp = L_out[worldid, Madr_ki] / L_out[worldid, diag_k]
  for j in range(M_rownnz[i]):
    wp.atomic_sub(L_out[worldid], Madr_i + j, L_in[worldid, M_rowadr[k] + j] * tmp)  # kernel_analyzer: ignore[atomic]
  L_out[worldid, Madr_ki] = tmp


@wp.kernel
def _qLDiag_div_reference(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # In:
  L_in: wp.array2d[float],
  # Out:
  D_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  diag_i = M_rowadr[dofid] + M_rownnz[dofid] - 1
  D_out[worldid, dofid] = 1.0 / L_in[worldid, diag_i]


def _factor_reference(m, d, M, L, D):
  wp.copy(L, M)
  for updates in reversed(m.qLD_updates):
    wp.launch(_qLD_acc_reference, dim=(d.nworld, updates.size), inputs=[m.M_rownnz, m.M_rowadr, updates, L], outputs=[L])
  wp.launch(_qLDiag_div_reference, dim=(d.nworld, m.nv), inputs=[m.M_rownnz, m.M_rowadr, L], outputs=[D])


def _fixture(nworld: int = 3):
  mjm, mjd, m, d = test_data.fixture(xml=_tree_xml(), nworld=nworld)
  # distinct configurations per world, so every world factors a different M
  rng = np.random.default_rng(0)
  qpos = np.tile(mjm.qpos0, (nworld, 1))
  qpos[:, 7:] += rng.uniform(-1.0, 1.0, size=(nworld, mjm.nq - 7))
  qpos[:, 3:7] = rng.normal(size=(nworld, 4))
  qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
  wp.copy(d.qpos, wp.array(qpos.astype(np.float32), dtype=float))
  mjw.fwd_position(m, d, factorize=False)
  return mjm, m, d


def _assert_same_class(actual, expected, name):
  """Bitwise where the update order is serial (CPU); rounding-level where GPU atomics reorder.

  On the GPU the old kernels themselves differ run to run (atomics within a level): measured on this
  fixture, old vs old and new vs old both stay ~30x inside the bound below.
  """
  if wp.get_device().is_cuda:
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5 * np.abs(expected).max(), err_msg=name)
  else:
    np.testing.assert_array_equal(actual, expected, err_msg=name)


def _assert_solves(mjm, M, x, y):
  """Componentwise residual of M x = y at float32 rounding level (backward stable)."""
  for w in range(M.shape[0]):
    dense = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sym2dense(dense, M[w].astype(np.float64), mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    xw, yw = x[w].astype(np.float64), y[w].astype(np.float64)
    np.testing.assert_array_less(np.abs(dense @ xw - yw), 1e-4 * (np.abs(dense) @ np.abs(xw)) + 1e-6)


class SparseLDLTest(parameterized.TestCase):
  def test_model_layout(self):
    """The element list covers every (update, j) pair of each level, in serial order."""
    mjm, m, _ = _fixture(nworld=1)
    self.assertGreater(mjm.nv, types.M_BLOCK_DENSE_MAX)
    updates = m.qLD_all_updates.numpy()
    offsets = m.qLD_level_offsets.numpy()
    elems = m.qLD_factor_elem.numpy()
    elem_offsets = m.qLD_factor_elem_offsets.numpy()
    self.assertEqual(len(elem_offsets), len(offsets))
    self.assertGreater(len(offsets) - 1, 20)
    rowadr, rownnz = mjm.M_rowadr, mjm.M_rownnz
    for level in range(len(offsets) - 1):
      np.testing.assert_array_equal(updates[offsets[level] : offsets[level + 1]], m.qLD_updates[level].numpy())
      expected = [
        (rowadr[i] + j, rowadr[k] + j, Madr_ki, rowadr[k] + rownnz[k] - 1)
        for i, k, Madr_ki in updates[offsets[level] : offsets[level + 1]]
        for j in range(rownnz[i])
      ]
      np.testing.assert_array_equal(elems[elem_offsets[level] : elem_offsets[level + 1]], expected)
    self.assertEqual(elem_offsets[-1], len(elems))

  @parameterized.parameters(1, 32, 256, 1024)
  def test_factor_matches_level_kernels(self, block_dim):
    """The fused kernel equals the original per-level kernels."""
    mjm, m, d = _fixture()
    m.block_dim.factor_i_sparse_fused = block_dim
    L_region = d.qLD[:, m.qLD_block_total :]
    smooth._factor_i_sparse(m, d, d.M, L_region, d.qLDiagInv)
    L_new, D_new = L_region.numpy(), d.qLDiagInv.numpy()

    L_ref = wp.empty_like(L_region)
    D_ref = wp.empty_like(d.qLDiagInv)
    _factor_reference(m, d, d.M, L_ref, D_ref)

    _assert_same_class(L_new, L_ref.numpy(), "qLD")
    _assert_same_class(D_new, D_ref.numpy(), "qLDiagInv")

    # and the sparse-block rows are MuJoCo's factor for the same configuration
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = d.qpos.numpy()[1]
    mujoco.mj_forward(mjm, mjd)
    sparse = m.qLD_block_adr.numpy() == types.Q_LD_BLOCK_SPARSE
    rows = [np.arange(mjm.M_rowadr[i], mjm.M_rowadr[i] + mjm.M_rownnz[i]) for i in np.flatnonzero(sparse)]
    adr = np.concatenate(rows)
    np.testing.assert_allclose(L_new[1, adr], mjd.qLD[adr], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(D_new[1, sparse], mjd.qLDiagInv[sparse], rtol=1e-3, atol=1e-3)

  @parameterized.parameters("option", "warp_config")
  def test_deterministic_factor_and_solve(self, source):
    """Deterministic atomics take the per-level paths and still factor and solve M.

    Requested by Option.deterministic, or by Warp's global configuration for kernels created in it.
    """
    mjm, m, d = _fixture()
    rng = np.random.default_rng(1)
    y = wp.array(rng.normal(size=(d.nworld, m.nv)).astype(np.float32), dtype=float)
    mjw.factor_m(m, d)
    L_ref, D_ref = d.qLD.numpy(), d.qLDiagInv.numpy()

    saved = wp.config.deterministic
    if source == "option":
      m.opt.deterministic = types.DeterminismType.ATOMICS
    else:
      wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
      warp_util._KERNEL_CACHE.clear()
    try:
      fused = smooth._factor_i_sparse_fused(m.nv, len(m.qLD_updates))
      self.assertEqual(smooth._deterministic_launch(fused), source == "warp_config")
      d.qLD.zero_()
      mjw.factor_m(m, d)
      x = wp.zeros((d.nworld, m.nv), dtype=float)
      mjw.solve_m(m, d, x, y)
      L_det, D_det, x_det = d.qLD.numpy(), d.qLDiagInv.numpy(), x.numpy()
    finally:
      wp.config.deterministic = saved
      m.opt.deterministic = types.DeterminismType.NONE
      warp_util._KERNEL_CACHE.clear()

    sparse = slice(m.qLD_block_total, None)
    np.testing.assert_allclose(L_det[:, sparse], L_ref[:, sparse], rtol=1e-4, atol=1e-5 * np.abs(L_ref).max())
    np.testing.assert_allclose(D_det, D_ref, rtol=1e-4, atol=1e-5 * np.abs(D_ref).max())
    _assert_solves(mjm, d.M.numpy(), x_det, y.numpy())

  @parameterized.parameters(32, 512, 1024)
  def test_solve_block_dims(self, block_dim):
    """The backsubstitution gives the same x at any block width; it solves M x = y."""
    mjm, m, d = _fixture()
    mjw.factor_m(m, d)
    rng = np.random.default_rng(1)
    y = wp.array(rng.normal(size=(d.nworld, m.nv)).astype(np.float32), dtype=float)

    results = {}
    for dim in (128, block_dim):
      m.block_dim.solve_LD_sparse_fused = dim
      x = wp.zeros((d.nworld, m.nv), dtype=float)
      mjw.solve_m(m, d, x, y)
      results[dim] = x.numpy()
    _assert_same_class(results[block_dim], results[128], "x")
    _assert_solves(mjm, d.M.numpy(), results[block_dim], y.numpy())


if __name__ == "__main__":
  wp.init()
  absltest.main()
