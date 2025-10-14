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

enum class EStyle3DCustomBit : mjtByte
{
	Default = 1 << 0,
	SkipSim = 1 << 1,
	SetPos = 1 << 2,
	ClearVel = 1 << 3,
	ClearPin = 1 << 4,
	RestorePin = 1 << 5,
};

// TODO: make it safer 
class Style3DSimHndManager
{
public:

	Style3DSimHndManager() {}

	~Style3DSimHndManager() { Clear(); }

	static Style3DSimHndManager& GetSingleton()
	{
		static Style3DSimHndManager manager;
		return manager;
	}

	void Clear()
	{
		masterInstance = -1;

		SrWorld_Destroy(&worldHnd);
		for (auto& hnd : colliderHnds)
		{
			SrMeshCollider_Destroy(&hnd);
		}
		for (auto& hnd : rigidHnds)
		{
			SrRigidBody_Destroy(&hnd);
		}
		colliderHnds.clear();
		for (auto& hnd : meshHnds)
		{
			SrMesh_Destroy(&hnd.second);
		}
		meshHnds.clear();
		for (auto& hnd : clothHnds)
		{
			SrCloth_Destroy(&hnd.second);
		}
		clothHnds.clear();
		geoTransforms.clear();
		geoSlideFrictions.clear();
		geoMeshIndexPair.clear();
	}

public:

	int masterInstance = -1;

	SrWorldHnd worldHnd = nullptr;
	std::vector<SrMeshColliderHnd> colliderHnds;
	std::vector<SrRigidBodyHnd> rigidHnds;
	std::map<int, SrMeshHnd> meshHnds;
	std::map<int, SrClothHnd> clothHnds; // key is flex id
	std::map<int, SrTransform> geoTransforms;
	std::map<int, mjtNum> geoSlideFrictions;
	std::vector<SrVec2i> geoMeshIndexPair; // collider geo mesh
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

  Style3DSim(const mjModel* m, mjData* d, int instance);

  void CreateS3dMeshes(const mjModel* m, mjData* d);

  void InitFlexIdx(const mjModel* m, mjData* d, int instance);

 private:

  int flexIdx = -1;
  
  bool keepWrinkles = false;
  double solidifyStiff = 0.0;
  std::vector<int> pinVerts;
  SrClothSimAttribute clothSimAttribute;

  // global settings, only use master data
  int substep = 1;
  bool useRigidCollider = false;
  bool useConvexHull = false;
  std::string usr = "";
  std::string pwd = "";
  SrColliderSimAttribute colliderSimAttribute;
  SrWorldSimAttribute worldSimAttribute;
};

}  // namespace mujoco::plugin::style3dsim

#endif  // MUJOCO_SRC_PLUGIN_STYLE3DSIM_H_
