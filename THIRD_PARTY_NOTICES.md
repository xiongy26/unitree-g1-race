# 第三方组件与资产许可声明（THIRD_PARTY_NOTICES）

本仓库（g1-race）自身代码（根目录脚本、`src/`、`index.html`、`tools/`、`server.js` 等）
以 **MIT 许可证** 发布，见 [LICENSE](LICENSE)。

`vendor/` 与 `assets/` 中包含的第三方组件与机器人模型/策略资产，按其各自的上游许可证
授权，与上游仓库的精确映射如下文。各许可证全文与上游原文摘录见文末附录。

## 一、vendor/ —— 本地化的前端依赖

| 路径 | 组件与版本 | 许可证 | 上游 |
|---|---|---|---|
| `vendor/mujoco/` | MuJoCo 官方 WebAssembly 绑定（npm 包 `@mujoco/mujoco`） | Apache-2.0 | <https://github.com/google-deepmind/mujoco> |
| `vendor/ort/` | onnxruntime-web 1.19.2（仅 WASM CPU 后端所需文件） | MIT | <https://github.com/microsoft/onnxruntime>（onnxruntime-web） |
| `vendor/three/three.module.js` | three.js r160 | MIT | <https://github.com/mrdoob/three.js> |
| `vendor/three/controls/OrbitControls.js` | three.js r160 官方控件（无逐文件许可头，按 three.js 仓库整体 MIT 授权） | MIT | 同上 |

> 说明：为控制仓库体积，`vendor/ort/` 仅保留 WASM CPU 后端加载所需的
> `ort.min.js`、`ort-wasm-simd-threaded.mjs`、`ort-wasm-simd-threaded.wasm`；
> 面向 WebGPU/WebNN 执行后端的 jsep 变体（`ort-wasm-simd-threaded.jsep.wasm` 与
> `ort-wasm-simd-threaded.jsep.mjs`，约 21MB）经核实在本项目的运行路径中不会被加载，
> 已从本仓库移除。如需完整发行版请从上游获取。

## 二、assets/ —— 机器人模型与策略权重

| 资产目录 | 物种 | 模型上游（许可证） | 策略权重上游（许可证） |
|---|---|---|---|
| `assets/g1_29dof.xml`、`assets/meshes/`、`assets/policy.onnx` | Unitree G1 | [unitree_ros](https://github.com/unitreerobotics/unitree_ros)（BSD-3-Clause） | [RoboCubPilot/g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco)（**上游未声明许可证**，见下文说明） |
| `assets/pm01/` | 众擎 PM01 | [engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab)（BSD-3-Clause） | 同模型上游（官方 AMP 速度策略 `model_19999.onnx`） |
| `assets/t1/` | Booster T1 | [booster_gym](https://github.com/BoosterRobotics/booster_gym)（Apache-2.0 + 附加归属段） | 同模型上游（官方 `T1.pt`，本仓库转 ONNX） |
| `assets/tk/` | 天工 Tienkung2-Lite | [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)（自定义许可：BSD-3-Clause 系多段归属，见附录六） | 同模型上游（官方 `Exported_policy/walk.pt`，本仓库转 ONNX） |
| `assets/x1/` | 智元灵犀 X1 | [agibot_x1_infer](https://github.com/AgibotTech/agibot_x1_infer)（Mulan PSL v2） | 同模型上游（官方 `rl_walk_leg_shoulder.onnx` 与 `rl_walk_leg.onnx`） |
| `assets/duck/` | Pollen MicroDuck | [microduck_rl](https://github.com/pollen-robotics/microduck_rl)（Apache-2.0） | HuggingFace [microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)（Apache-2.0） |

### 1. Unitree G1（`assets/g1_29dof.xml`、`assets/meshes/`、`assets/policy.onnx`）

- 机器人模型（MJCF + STL 网格）：改编自
  [unitree_ros](https://github.com/unitreerobotics/unitree_ros)（BSD-3-Clause），
  建模过程另参考 [loco-lab](https://github.com/JacobEGarcia/loco-lab)（许可情况见其上游仓库）。
- 策略权重（`assets/policy.onnx`）：来自
  [RoboCubPilot/g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco)。
  **该上游仓库未声明任何许可证**（GitHub 许可识别为 None）。本仓库仅出于学习研究目的
  随源再分发该权重，其使用同样仅限学习研究；如您是权利人或对该权重的分发有疑虑，
  请通过 issue 联系，我们将及时处理。

### 2. 众擎 PM01（`assets/pm01/`）

- 模型与策略均来自 [engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab)
  （BSD-3-Clause）：官方 MJCF + 网格，以及官方 AMP 速度策略 `model_19999.onnx`
  （本仓库更名为 `policy.onnx`）。

### 3. Booster T1（`assets/t1/`）

- 模型与权重均来自 [booster_gym](https://github.com/BoosterRobotics/booster_gym)
  （Apache-2.0；上游 LICENSE 含版权行与归属段，逐字摘录见附录五）：官方
  `T1_locomotion.xml` + 网格；官方 TorchScript 权重 `T1.pt`（**仅溯源保留，
  运行时不加载**）由本仓库转换为 `policy.onnx`。

### 4. 天工 Tienkung2-Lite（`assets/tk/`）

- 模型与权重均来自 [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)：
  官方 MJCF（`tienkung2_lite`）+ 网格；官方 TorchScript 权重
  `Exported_policy/walk.pt`（**仅溯源保留，运行时不加载**）由本仓库转换为 `policy.onnx`。
- 上游 LICENSE 为自定义许可文本：由 RSL-RL、Isaac Lab、Legged Lab、TienKung-Lab
  四段版权归属声明与 BSD-3-Clause 条款构成，摘录见附录六。

### 5. 智元灵犀 X1（`assets/x1/`）

- 模型与策略均来自 [agibot_x1_infer](https://github.com/AgibotTech/agibot_x1_infer)
  （Mulan PSL v2，全文见附录四）：官方 serial MJCF + 网格，以及官方行走策略
  `rl_walk_leg_shoulder.onnx`（现用，本仓库更名为 `policy_shoulder.onnx`）与
  `rl_walk_leg.onnx`（`policy.onnx`，备用可随时切回）。

### 6. Pollen MicroDuck（`assets/duck/`）

- 模型来自 [microduck_rl](https://github.com/pollen-robotics/microduck_rl)
  （Apache-2.0）：官方 VelStand 任务训练模型 `scene_allcollisions.xml` +
  `robot_allcollisions.xml` + 官方网格（其中 12 个大网格经本仓库抽稀至 4000 三角面
  以适配 MuJoCo WASM 内存上限，见 README「已知实现要点」）。
- 策略来自 HuggingFace [microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)
  （Apache-2.0）：官方 `velstand.onnx`（本仓库更名为 `policy.onnx`）。
- 致谢：执行器等效性验证使用了 [Rhoban/bam](https://github.com/Rhoban/bam)
  的 XL330 m6 模型参数（仅验证用途，未随本仓库分发）。

## 三、其他

- `assets/meshlist.txt`：G1 模型的网格文件清单（本仓库生成的辅助文件，供排查网格缺失，
  非第三方资产）。

---

## 附录一：MIT 许可证（本项目代码；onnxruntime-web、three.js 同用此许可证）

标准文本，官方模板见 <https://opensource.org/licenses/MIT>。

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

本项目代码的版权声明为 `Copyright (c) 2026 xiongy26`，见 [LICENSE](LICENSE)。

## 附录二：BSD-3-Clause 许可证（unitree_ros、engineai_rl_lab 使用；TienKung-Lab 归属段所引条款同此文本）

标准文本，官方模板见 <https://opensource.org/licenses/BSD-3-Clause>。

```text
Copyright <YEAR> <COPYRIGHT HOLDER>

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 附录三：Apache License 2.0（@mujoco/mujoco、booster_gym、microduck_rl、microduck-policies 使用）

标准文本，官方页面见 <https://www.apache.org/licenses/LICENSE-2.0>。

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## 附录四：Mulan Permissive Software License，Version 2（木兰宽松许可证第2版，Mulan PSL v2，agibot_x1_infer 使用）

以下为 agibot_x1_infer 仓库分发的许可证原文（Mulan PSL v2），获取日期 2026-10-05
（官方发布页见 <http://license.coscl.org.cn/MulanPSL2>）。

```text
Mulan Permissive Software License，Version 2

Mulan Permissive Software License，Version 2 (Mulan PSL v2)

January 2020 http://license.coscl.org.cn/MulanPSL2

Your reproduction, use, modification and distribution of the Software shall be subject to Mulan PSL v2 (this License) with the following terms and conditions:

0. Definition

Software means the program and related documents which are licensed under this License and comprise all Contribution(s).

Contribution means the copyrightable work licensed by a particular Contributor under this License.

Contributor means the Individual or Legal Entity who licenses its copyrightable work under this License.

Legal Entity means the entity making a Contribution and all its Affiliates.

Affiliates means entities that control, are controlled by, or are under common control with the acting entity under this License, ‘control’ means direct or indirect ownership of at least fifty percent (50%) of the voting power, capital or other securities of controlled or commonly controlled entity.

1. Grant of Copyright License

Subject to the terms and conditions of this License, each Contributor hereby grants to you a perpetual, worldwide, royalty-free, non-exclusive, irrevocable copyright license to reproduce, use, modify, or distribute its Contribution, with modification or not.

2. Grant of Patent License

Subject to the terms and conditions of this License, each Contributor hereby grants to you a perpetual, worldwide, royalty-free, non-exclusive, irrevocable (except for revocation under this Section) patent license to make, have made, use, offer for sale, sell, import or otherwise transfer its Contribution, where such patent license is only limited to the patent claims owned or controlled by such Contributor now or in future which will be necessarily infringed by its Contribution alone, or by combination of the Contribution with the Software to which the Contribution was contributed. The patent license shall not apply to any modification of the Contribution, and any other combination which includes the Contribution. If you or your Affiliates directly or indirectly institute patent litigation (including a cross claim or counterclaim in a litigation) or other patent enforcement activities against any individual or entity by alleging that the Software or any Contribution in it infringes patents, then any patent license granted to you under this License for the Software shall terminate as of the date such litigation or activity is filed or taken.

3. No Trademark License

No trademark license is granted to use the trade names, trademarks, service marks, or product names of Contributor, except as required to fulfill notice requirements in section 4.

4. Distribution Restriction

You may distribute the Software in any medium with or without modification, whether in source or executable forms, provided that you provide recipients with a copy of this License and retain copyright, patent, trademark and disclaimer statements in the Software.

5. Disclaimer of Warranty and Limitation of Liability

THE SOFTWARE AND CONTRIBUTION IN IT ARE PROVIDED WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED. IN NO EVENT SHALL ANY CONTRIBUTOR OR COPYRIGHT HOLDER BE LIABLE TO YOU FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO ANY DIRECT, OR INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM YOUR USE OR INABILITY TO USE THE SOFTWARE OR THE CONTRIBUTION IN IT, NO MATTER HOW IT’S CAUSED OR BASED ON WHICH LEGAL THEORY, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

6. Language

THIS LICENSE IS WRITTEN IN BOTH CHINESE AND ENGLISH, AND THE CHINESE VERSION AND ENGLISH VERSION SHALL HAVE THE SAME LEGAL EFFECT. IN THE CASE OF DIVERGENCE BETWEEN THE CHINESE AND ENGLISH VERSIONS, THE CHINESE VERSION SHALL PREVAIL.

END OF THE TERMS AND CONDITIONS

How to Apply the Mulan Permissive Software License，Version 2 (Mulan PSL v2) to Your Software

To apply the Mulan PSL v2 to your work, for easy identification by recipients, you are suggested to complete following three steps:

Fill in the blanks in following statement, including insert your software name, the year of the first publication of your software, and your name identified as the copyright owner;

Create a file named "LICENSE" which contains the whole context of this License in the first directory of your software package;

Attach the statement to the appropriate annotated syntax at the beginning of each source file.

Copyright (c) [Year] [name of copyright holder]
[Software Name] is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
```

## 附录五：booster_gym LICENSE 归属段摘录（逐字）

来源：<https://github.com/BoosterRobotics/booster_gym/blob/main/LICENSE>
（获取日期 2026-10-05；完整 LICENSE = 以下版权行 + Apache-2.0 全文 + 归属说明段）。

```text
Copyright [2024] [Booster Robotics Technology Co., Ltd ("Booster Robotics")]

Licensed under the Apache License, Version 2.0 (the "License");
...
```

归属说明段（上游 LICENSE 中声明其代码基于以下项目构建）：

- <https://github.com/isaac-sim/IsaacGymEnvs>
- <https://github.com/leggedrobotics/legged_gym>
- <https://github.com/leggedrobotics/rsl_rl>
- <https://github.com/roboterax/humanoid-gym>

## 附录六：TienKung-Lab LICENSE 版权归属段摘录（逐字）

来源：<https://github.com/Open-X-Humanoid/TienKung-Lab/blob/main/LICENSE>
（获取日期 2026-10-05）。上游 LICENSE 由以下四段版权归属声明与 BSD-3-Clause
条款文本（同附录二）构成，此处逐字摘录归属声明部分，条款全文以上游为准：

```text
Copyright (c) 2021-2024, The RSL-RL Project Developers.
All rights reserved.
Original code is licensed under BSD-3-Clause.

Copyright (c) 2022-2025, The Isaac Lab Project Developers.
All rights reserved.
Original code is licensed under BSD-3-Clause.

Copyright (c) 2025-2026, The Legged Lab Project Developers.
All rights reserved.
Modifications are licensed under BSD-3-Clause.

Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
All rights reserved.
Modifications are licensed under BSD-3-Clause.
```
