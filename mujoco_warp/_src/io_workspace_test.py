# Copyright 2026 The MuJoCo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

import mujoco
import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjwarp
from mujoco_warp._src import io
from mujoco_warp._src import smooth
from mujoco_warp._src import types
from mujoco_warp._src import warp_util


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_const0_workspace_matches_allocating_path(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA is unavailable")
  mjm = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><site name="a" pos="0 0 2"/>
      <body pos="0 0 1"><freejoint/><geom size="0.1" mass="1"/>
        <site name="b" pos="0 0 0.1"/>
      </body></worldbody>
      <tendon><spatial name="stem"><site site="a"/><site site="b"/></spatial></tendon>
      <actuator><motor tendon="stem"/></actuator>
    </mujoco>""")
  with wp.ScopedDevice(device):
    m = mjwarp.put_model(mjm, batch_sizes={"dof_armature": 2, "dof_invweight0": 2, "body_invweight0": 2})
    d = mjwarp.make_data(mjm, nworld=2)
    workspace = mjwarp.Const0Workspace(m, d)
    m.dof_armature.assign(np.array([[0, 0, 0, 0.0014435225, 0.0014435225, 0.0014435225], [0] * 6], dtype=np.float32))
    mjwarp.set_const_0(m, d)
    expected = {
      name: getattr(m, name).numpy().copy()
      for name in ("dof_invweight0", "body_invweight0", "tendon_invweight0", "actuator_acc0")
    }
    # Warm all kernels before forbidding allocation. Repeated scratch reuse must
    # not depend on its previous contents or allocate nested tendon/transmission scratch.
    mjwarp.set_const_0(m, d, workspace=workspace)
    with (
      mock.patch.object(wp, "empty", side_effect=AssertionError("allocation")),
      mock.patch.object(wp, "zeros", side_effect=AssertionError("allocation")),
      mock.patch.object(wp, "clone", side_effect=AssertionError("allocation")),
    ):
      mjwarp.set_const_0(m, d, workspace=workspace)
    for name, value in expected.items():
      np.testing.assert_array_equal(getattr(m, name).numpy(), value, err_msg=name)


def _layout_coverage_xml() -> str:
  """Every inertia factor layout plus a tendon and joint/tendon actuators.

  A sparse-LDL tree (> M_BLOCK_DENSE_MAX DOFs), a tile tree, a scalar free body and a compact
  slide body.
  """

  def branch(name: str, depth: int, fork_every: int) -> str:
    if depth == 0:
      return ""
    joints = "".join(f'<joint name="{name}_{axis}" type="hinge" axis="{vec}" armature="0.01"/>' for axis, vec in _AXES)
    side = branch(name + "s", depth // 2, fork_every) if depth % fork_every == 0 else ""
    return (
      f'<body name="{name}" pos="0.01 0 0.1">{joints}<geom type="capsule" fromto="0 0 0 0 0.02 0.1" size="0.02" mass="0.2"/>'
      f"{side}{branch(name + 'c', depth - 1, fork_every)}</body>"
    )

  return f"""
    <mujoco><worldbody><site name="anchor" pos="0.3 0 1"/>
      {branch("a", 18, 4)}
      <body name="mid" pos="1 0 0">{branch("b", 4, 99)}</body>
      <body name="free" pos="0 1 1"><freejoint/><geom size="0.1" mass="1" pos="0.02 0 0"/>
        <site name="tip" pos="0 0 0.1"/></body>
      <body name="slider" pos="0 -1 1"><joint name="slide" type="slide" axis="1 0 0"/><geom size="0.05" mass="0.3"/></body>
    </worldbody>
    <tendon><spatial name="stem"><site site="anchor"/><site site="tip"/></spatial></tendon>
    <actuator><motor tendon="stem"/><motor joint="a_x"/><motor joint="slide"/></actuator>
    </mujoco>"""


_AXES = (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1"))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("rhs_chunk", [1, 7, None])
def test_const0_batched_inverse_weights_match_mujoco(device, rhs_chunk):
  """Batched inverse-weight solves agree with MuJoCo for every inertia factor layout."""
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA is unavailable")
  mjm = mujoco.MjModel.from_xml_string(_layout_coverage_xml())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_setConst(mjm, mjd)
  with wp.ScopedDevice(device):
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm, nworld=2)
    layout = io.m_block_layout(mjm)
    assert layout["has_sparse"] and layout["scalar_tiles"] and layout["gather_tiles"]
    assert np.any(layout["dof_adr"] == types.Q_LD_BLOCK_COMPACT)
    for name in ("dof_invweight0", "body_invweight0", "tendon_invweight0", "actuator_acc0"):
      getattr(m, name).zero_()
    workspace = mjwarp.Const0Workspace(m, d, rhs_chunk=rhs_chunk)
    mjwarp.set_const_0(m, d, workspace=workspace)
    # MuJoCo and MJWarp average a one-DOF body's translational weight differently (independent
    # of batching); compare every other body.
    bodies = np.arange(mjm.nbody) != mjm.body("slider").id
    for name in ("dof_invweight0", "body_invweight0", "tendon_invweight0", "actuator_acc0"):
      actual = getattr(m, name).numpy()
      expected = np.broadcast_to(getattr(mjm, name).reshape(1, *actual.shape[1:]), actual.shape)
      if name == "body_invweight0":
        actual, expected = actual[:, bodies], expected[:, bodies]
      np.testing.assert_allclose(actual, expected, rtol=1e-3, err_msg=name)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("source", ["option", "warp_config"])
def test_const0_batched_deterministic(device, source):
  """Deterministic batched solves (per-level launches) agree with the default path, run to run."""
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA is unavailable")
  mjm = mujoco.MjModel.from_xml_string(_layout_coverage_xml())
  names = ("dof_invweight0", "body_invweight0", "tendon_invweight0", "actuator_acc0")
  with wp.ScopedDevice(device):
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm, nworld=2)
    workspace = mjwarp.Const0Workspace(m, d, rhs_chunk=7)
    mjwarp.set_const_0(m, d, workspace=workspace)
    expected = {name: getattr(m, name).numpy().copy() for name in names}

    saved = wp.config.deterministic
    if source == "option":
      m.opt.deterministic = types.DeterminismType.ATOMICS
    else:
      wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
      warp_util._KERNEL_CACHE.clear()
    fused = smooth._solve_LD_sparse_fused_batched(m.nv, len(m.qLD_updates))
    assert smooth._deterministic_launch(fused) == (source == "warp_config")
    runs = []
    try:
      for _ in range(2):
        for name in names:
          getattr(m, name).zero_()
        mjwarp.set_const_0(m, d, workspace=workspace)
        runs.append({name: getattr(m, name).numpy().copy() for name in names})
    finally:
      wp.config.deterministic = saved
      m.opt.deterministic = types.DeterminismType.NONE
      warp_util._KERNEL_CACHE.clear()

  for name in names:
    # summation order differs from the default path (determinism_test's NONE vs ATOMICS tolerance)
    np.testing.assert_allclose(runs[0][name], expected[name], rtol=1e-3, atol=1e-3, err_msg=name)
    assert runs[0][name].tobytes() == runs[1][name].tobytes(), name
