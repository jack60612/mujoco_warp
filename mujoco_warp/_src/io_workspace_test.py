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
      name: getattr(m, name).numpy() for name in ("dof_invweight0", "body_invweight0", "tendon_invweight0", "actuator_acc0")
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
