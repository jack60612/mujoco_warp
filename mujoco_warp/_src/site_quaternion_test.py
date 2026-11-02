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

"""Regression tests for rotational site transmissions with reference sites."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data


class SiteQuaternionTest(parameterized.TestCase):
  @parameterized.named_parameters(("coincident_frames", None), ("rotated_frames", 0))
  def test_rotated_site_and_reference(self, keyframe):
    """Composes both local site rotations after their parent body rotations."""
    mjm, mjd, m, d = test_data.fixture("actuation/site_quaternion.xml", keyframe=keyframe, nworld=2)

    # Both sites and both parent bodies have noncommuting rotations.  Three
    # independent rotational gears expose every component of the relative rotation.
    # Use native world-site matrices as the independent geometric oracle:
    # MuJoCo 3.12's actuator_length has the same reversed-product bug.
    reference = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "reference")
    target = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "target")
    reference_quat = np.zeros(4)
    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(reference_quat, mjd.site_xmat[reference])
    mujoco.mju_mat2Quat(target_quat, mjd.site_xmat[target])
    relative_rotation = np.zeros(3)
    mujoco.mju_subQuat(relative_rotation, target_quat, reference_quat)
    expected_length = mjm.actuator_gear[:, 3:] @ relative_rotation
    if keyframe is None:
      # Ry(90) Rz(90) = Rx(90) Ry(90): different body/local
      # factorizations describe identical authored world-site orientations.
      np.testing.assert_allclose(mjd.site_xmat[target], mjd.site_xmat[reference], atol=1e-12, rtol=0)
      np.testing.assert_allclose(expected_length, 0, atol=1e-12, rtol=0)

    d.actuator_length.fill_(wp.inf)
    d.actuator_moment.fill_(wp.inf)
    mjw.transmission(m, d)

    expected_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(
      expected_moment, mjd.actuator_moment, mjd.moment_rownnz, mjd.moment_rowadr, mjd.moment_colind
    )
    for world in range(d.nworld):
      np.testing.assert_allclose(d.actuator_length.numpy()[world], expected_length, atol=5e-5, rtol=5e-5)
      moment = np.zeros((mjm.nu, mjm.nv))
      mujoco.mju_sparse2dense(
        moment,
        d.actuator_moment.numpy()[world],
        d.moment_rownnz.numpy()[world],
        d.moment_rowadr.numpy()[world],
        d.moment_colind.numpy()[world],
      )
      np.testing.assert_allclose(moment, expected_moment, atol=5e-5, rtol=5e-5)


if __name__ == "__main__":
  absltest.main()
