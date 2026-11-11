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

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import derivative
from mujoco_warp._src import io
from mujoco_warp._src.types import DeterminismType

# A branch (hinge, ball, hinge) with two sibling twigs and two free fruit hung from the twigs by
# stiff, damped spatial tendons: every stem couples a free body to a chain of the branch tree. A
# damped fixed tendon couples the sibling twigs (same tree, outside M's ancestor sparsity) and a
# site actuator with a velocity bias couples the branch tip to a reference site on the other twig.
# `{fruit}` and `{tree}` let a test declare the fruit before the branch (dof order) or after it.
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
      <geom type="sphere" size=".035" mass=".15"/>
      <site name="stem_a" pos="0 0 .035"/>
    </body>
    <body name="fruit_b" pos=".6 -.05 .9">
      <freejoint/>
      <geom type="ellipsoid" size=".03 .035 .04" mass=".12"/>
      <site name="stem_b" pos=".005 0 .04"/>
    </body>
"""
_XML = """
<mujoco>
  <option integrator="discrete" timestep="{dt}"/>
  <default><geom type="capsule" size=".01" contype="0" conaffinity="0"/></default>
  <worldbody>
    {first}
    {second}
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


def _xml(fruit_first: bool, dt: float = 0.002) -> str:
  tree_first = not fruit_first
  return _XML.format(
    dt=dt,
    first=_TREE if tree_first else _FRUIT,
    second=_FRUIT if tree_first else _TREE,
    qpos=f"{_TREE_QPOS}  {_FRUIT_QPOS}" if tree_first else f"{_FRUIT_QPOS}  {_TREE_QPOS}",
    qvel=f"{_TREE_QVEL}  {_FRUIT_QVEL}" if tree_first else f"{_FRUIT_QVEL}  {_TREE_QVEL}",
  )


def _metric_residual(m, d) -> float:
  """|M_hat qacc_smooth - f| / |f| with the matrix-free metric product."""
  rhs = wp.zeros_like(d.qacc_smooth)
  wp.launch(
    derivative._eff_rhs,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.enableflags, m.dof_treeid, d.tree_awake, d.qfrc_smooth, d.efm_c, d.efm_ca],
    outputs=[rhs],
  )
  prod = wp.zeros_like(d.qacc_smooth)
  derivative.eff_mul_m(m, d, prod, d.qacc_smooth)
  rhs, prod = rhs.numpy(), prod.numpy()
  return float(np.max(np.linalg.norm(prod - rhs, axis=1) / np.linalg.norm(rhs, axis=1)))


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

  def test_deterministic_atomics(self):
    """The deterministic launches assemble and factor the same metric."""
    _, _, m, d = test_data.fixture(xml=_xml(False), keyframe=0)
    mjw.forward(m, d)
    expected = d.qacc_smooth.numpy()
    m.opt.deterministic = DeterminismType.ATOMICS
    d.qacc_smooth.zero_()
    mjw.forward(m, d)
    np.testing.assert_allclose(d.qacc_smooth.numpy(), expected, rtol=1e-5, atol=1e-6 * np.max(np.abs(expected)))

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


if __name__ == "__main__":
  wp.init()
  absltest.main()
