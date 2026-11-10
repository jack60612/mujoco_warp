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

from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import smooth
from mujoco_warp._src import types


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
    wp.atomic_sub(L_out[worldid], Madr_i + j, L_in[worldid, M_rowadr[k] + j] * tmp)
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
  offsets = m.qLD_level_offsets.numpy()
  for level in reversed(range(len(offsets) - 1)):
    updates = m.qLD_all_updates[int(offsets[level]) : int(offsets[level + 1])]
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
      expected = [
        (rowadr[i] + j, rowadr[k] + j, Madr_ki, rowadr[k] + rownnz[k] - 1)
        for i, k, Madr_ki in updates[offsets[level] : offsets[level + 1]]
        for j in range(rownnz[i])
      ]
      np.testing.assert_array_equal(elems[elem_offsets[level] : elem_offsets[level + 1]], expected)
    self.assertEqual(elem_offsets[-1], len(elems))
    self.assertEqual(m.qLD_level_offsets_host, tuple(int(x) for x in offsets))
    self.assertEqual(m.qLD_factor_elem_offsets_host, tuple(int(x) for x in elem_offsets))

  @parameterized.parameters((True, 1), (True, 32), (True, 256), (True, 1024), (False, 1024))
  def test_factor_matches_level_kernels(self, fused, block_dim):
    """The fused kernel and the per-level element launches equal the original per-level kernels."""
    mjm, m, d = _fixture()
    m.block_dim.factor_i_sparse_fused = block_dim
    L_region = d.qLD[:, m.qLD_block_total :]
    with mock.patch.object(smooth, "_factor_sparse_fused", return_value=fused):
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

  def test_factor_path_choice(self):
    """Fused unless levels are wide and most SMs idle; fused on CPU and in deterministic mode."""
    _, m, _ = _fixture(nworld=1)
    width = m.qLD_factor_elem_offsets_host[-1] / (len(m.qLD_level_offsets_host) - 1)
    self.assertLess(width, types.M_SPARSE_LEVEL_WIDTH_MIN)
    self.assertTrue(smooth._factor_sparse_fused(m, 1))

    device = wp.get_device()
    with mock.patch.object(smooth, "M_SPARSE_LEVEL_WIDTH_MIN", width / 2):
      if device.is_cuda:
        self.assertFalse(smooth._factor_sparse_fused(m, 1))
        self.assertTrue(smooth._factor_sparse_fused(m, device.sm_count // 2))
        # deterministic launches would reserve scatter records for every element thread
        with mock.patch.object(smooth, "_deterministic_records_per_thread", return_value=1):
          self.assertTrue(smooth._factor_sparse_fused(m, 1))
      else:
        self.assertTrue(smooth._factor_sparse_fused(m, 1))

  def test_factor_m_levels_captured(self):
    """factor_m on the per-level path replays from a CUDA graph like an eager call."""
    if not wp.get_device().is_cuda:
      self.skipTest("CUDA graph capture")
    _, m, d = _fixture()
    with mock.patch.object(smooth, "_factor_sparse_fused", return_value=False):
      mjw.factor_m(m, d)
      L_eager, D_eager = d.qLD.numpy(), d.qLDiagInv.numpy()
      d.qLD.zero_()
      d.qLDiagInv.zero_()
      with wp.ScopedCapture() as capture:
        mjw.factor_m(m, d)
    wp.capture_launch(capture.graph)
    _assert_same_class(d.qLD.numpy(), L_eager, "qLD")
    _assert_same_class(d.qLDiagInv.numpy(), D_eager, "qLDiagInv")

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

    # the full solve (sparse plus small-chain block) is backward stable: componentwise residual of
    # M x = y at float32 rounding level
    M = d.M.numpy()
    for w in range(d.nworld):
      dense = np.zeros((mjm.nv, mjm.nv))
      mujoco.mju_sym2dense(dense, M[w].astype(np.float64), mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
      xw, yw = results[block_dim][w].astype(np.float64), y.numpy()[w].astype(np.float64)
      np.testing.assert_array_less(np.abs(dense @ xw - yw), 1e-4 * (np.abs(dense) @ np.abs(xw)) + 1e-6)


if __name__ == "__main__":
  wp.init()
  absltest.main()
