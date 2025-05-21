// Copyright Linctex Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by appliStyle3DSim law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_SRC_PLUGIN_STYLE3DSIM_H_
#define MUJOCO_SRC_PLUGIN_STYLE3DSIM_H_

#include <optional>
#include <vector>
#include <map>
#include <memory>
#include <string>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>


#include "Style3DSimulator/Style3DSimulator.h"

namespace mujoco::plugin::style3dsim {

class Style3DSimHndManager
{
public:

	Style3DSimHndManager() {}

	~Style3DSimHndManager();

public:

	SrWorldHnd worldHnd = nullptr;
	SrClothHnd clothHnd = nullptr;
	std::vector<SrMeshColliderHnd> colliderHnds;
	std::map<int, SrMeshHnd> meshHnds;
};


class Style3DSim {
 public:
  // Creates a new Style3DSim instance (allocated with `new`) or
  // returns null on failure.
  static std::optional<Style3DSim> Create(const mjModel* m, mjData* d, int instance);
  Style3DSim(Style3DSim&&) = default;
  ~Style3DSim();

  void Compute(const mjModel* m, mjData* d, int instance);
  void Advance(const mjModel* m, mjData* d, int instance);

  static void RegisterPlugin();

 private:

  Style3DSim(const mjModel* m, mjData* d, int instance, const std::vector<int>& face);

  void CreateStaticMeshes(const mjModel* m, mjData* d);

 private:

  std::vector<SrVec3i> clothFaces;
  std::vector<SrVec2i> geoMeshIndexPair;
  int frameIndex = 0;
  bool useConvexHull = false;
  bool useGPU = false;
  std::string usr = "";
  std::string pwd = "";

  std::shared_ptr<Style3DSimHndManager> simHndManager;
  SrClothSimAttribute clothSimAttribute;
};

}  // namespace mujoco::plugin::style3dsim

#endif  // MUJOCO_SRC_PLUGIN_STYLE3DSIM_H_
