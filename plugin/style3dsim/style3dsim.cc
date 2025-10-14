// Copyright Linctex Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <optional>
#include <unordered_map>
#include <cstring>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include "style3dsim.h"

namespace mujoco::plugin::style3dsim {
namespace {

// reads numeric attributes
bool CheckNumAttr(const char* name, const mjModel* m, int instance) {
  char *end;
  std::string value = mj_getPluginConfig(m, instance, name);
  if (value.size() == 0)
	return false;
  value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
  strtod(value.c_str(), &end);
  return end == value.data() + value.size();
}

template<typename T>
void String2Vector(const std::string& txt, std::vector<T>& vec) {
	std::stringstream strm(txt);
	vec.clear();

	while (!strm.eof()) {
		T num;
		strm >> num;
		if (strm.fail()) {
			break;
		}
		else {
			vec.push_back(num);
		}
	}
}

bool IsEqual(const float& a, const float& b) {

	return std::abs(a - b) < 1.0e-6;
}

bool IsEqual(const SrTransform& a, const SrTransform& b) {

	if (!IsEqual(a.scale.x, b.scale.x)) return false;
	if (!IsEqual(a.scale.y, b.scale.y)) return false;
	if (!IsEqual(a.scale.z, b.scale.z)) return false;

	if (!IsEqual(a.translation.x, b.translation.x)) return false;
	if (!IsEqual(a.translation.y, b.translation.y)) return false;
	if (!IsEqual(a.translation.z, b.translation.z)) return false;

	if (!IsEqual(a.rotation.x, b.rotation.x)) return false;
	if (!IsEqual(a.rotation.y, b.rotation.y)) return false;
	if (!IsEqual(a.rotation.z, b.rotation.z)) return false;
	if (!IsEqual(a.rotation.w, b.rotation.w)) return false;

	return true;
}

}  // namespace


// factory function
std::optional<Style3DSim> Style3DSim::Create(
  const mjModel* m, mjData* d, int instance) {

	return Style3DSim(m, d, instance);
}

// plugin constructor
Style3DSim::Style3DSim(const mjModel* m, mjData* d, int instance) {

	InitFlexIdx(m, d, instance);

	//~ global settings

	// collider friction
	if (CheckNumAttr("friction", m, instance))
	{
		colliderSimAttribute.dynamicFriction = strtod(mj_getPluginConfig(m, instance, "friction"), nullptr);
	}

	if (CheckNumAttr("staticfriction", m, instance))
	{
		colliderSimAttribute.staticFriction = strtod(mj_getPluginConfig(m, instance, "staticfriction"), nullptr);
	}

	// collider gap
	if (CheckNumAttr("gap", m, instance))
	{
		colliderSimAttribute.CollisionGap = strtod(mj_getPluginConfig(m, instance, "gap"), nullptr) * 1.0e-3;
	}

	// ground friction
	if (CheckNumAttr("groundfriction", m, instance))
	{
		worldSimAttribute.groundDynamicFriction = strtod(mj_getPluginConfig(m, instance, "groundfriction"), nullptr);
	}

	if (CheckNumAttr("groundstaticfriction", m, instance))
	{
		worldSimAttribute.groundStaticFriction = strtod(mj_getPluginConfig(m, instance, "groundstaticfriction"), nullptr);
	}

	{
		const char* config = mj_getPluginConfig(m, instance, "convex");
		if (config)
		{
			std::string tmp = config;
			useConvexHull = tmp == "true";
		}
	}

	{
		const char* config = mj_getPluginConfig(m, instance, "rigidcollider");
		if (config)
		{
			std::string tmp = config;
			useRigidCollider = tmp == "true";
		}
	}

	{
		const char* config = mj_getPluginConfig(m, instance, "selfcollide");
		if (config)
		{
			std::string tmp = config;
			worldSimAttribute.enableSelfCollision = worldSimAttribute.enableUntangle = tmp == "true";
		}
	}

	if (CheckNumAttr("airdamping", m, instance))
	{
		worldSimAttribute.airDamping = strtod(mj_getPluginConfig(m, instance, "airdamping"), nullptr);
	}

	if (CheckNumAttr("stretchdamping", m, instance))
	{
		worldSimAttribute.stretchDamping = strtod(mj_getPluginConfig(m, instance, "stretchdamping"), nullptr);
	}

	if (CheckNumAttr("benddamping", m, instance))
	{
		worldSimAttribute.bendDamping = strtod(mj_getPluginConfig(m, instance, "benddamping"), nullptr);
	}

	if (CheckNumAttr("velsmoothing", m, instance))
	{
		worldSimAttribute.velSmoothing = strtod(mj_getPluginConfig(m, instance, "velsmoothing"), nullptr);
	}

	if (CheckNumAttr("groundheight", m, instance))
	{
		worldSimAttribute.groundHeight = strtod(mj_getPluginConfig(m, instance, "groundheight"), nullptr) * 1.0e-3;
	}

	{
		const char* config = mj_getPluginConfig(m, instance, "gpu");
		if (config)
		{
			std::string tmp = config;
			worldSimAttribute.enableGPU = tmp == "true";
		}
	}

	if (CheckNumAttr("substep", m, instance))
	{
		substep = strtod(mj_getPluginConfig(m, instance, "substep"), nullptr);
		substep = std::max(1, substep);
	}

	{
		const char* config = mj_getPluginConfig(m, instance, "user");
		if (config)
			usr = config;
	}

	{
		const char* config = mj_getPluginConfig(m, instance, "pwd");
		if (config)
			pwd = config;
	}


	//~ cloth settings

	//if (CheckNumAttr("stretch", m, instance))
	{
		std::vector<double> stiffVec;
		String2Vector(mj_getPluginConfig(m, instance, "stretch"), stiffVec);
		if (stiffVec.size() >= 1)
		{
			// init three value
			clothSimAttribute.stretchStiffness.x = stiffVec[0] * 1.0e-3;
			clothSimAttribute.stretchStiffness.y = clothSimAttribute.stretchStiffness.x;
			clothSimAttribute.stretchStiffness.z = clothSimAttribute.stretchStiffness.x;
		}

		if (stiffVec.size() >= 2)
		{
			clothSimAttribute.stretchStiffness.y = stiffVec[1] * 1.0e-3;
		}

		if (stiffVec.size() >= 3)
		{
			clothSimAttribute.stretchStiffness.z = stiffVec[2] * 1.0e-3;
		}
	}
	//if (CheckNumAttr("bend", m, instance))
	{
		std::vector<double> stiffVec;
		String2Vector(mj_getPluginConfig(m, instance, "bend"), stiffVec);
		if (stiffVec.size() >= 1)
		{
			// init three value
			clothSimAttribute.bendStiffness.x = stiffVec[0] * 1.0e-9;
			clothSimAttribute.bendStiffness.y = clothSimAttribute.bendStiffness.x;
			clothSimAttribute.bendStiffness.z = clothSimAttribute.bendStiffness.x;
		}

		if (stiffVec.size() >= 2)
		{
			clothSimAttribute.bendStiffness.y = stiffVec[1] * 1.0e-9;
		}

		if (stiffVec.size() >= 3)
		{
			clothSimAttribute.bendStiffness.z = stiffVec[2] * 1.0e-9;
		}
	}
	if (CheckNumAttr("thickness", m, instance))
	{
		clothSimAttribute.thickness = strtod(mj_getPluginConfig(m, instance, "thickness"), nullptr) * 1.0e-3;
	}
	if (CheckNumAttr("density", m, instance))
	{
		clothSimAttribute.density = strtod(mj_getPluginConfig(m, instance, "density"), nullptr) * 1.0e-3;
	}
	if (CheckNumAttr("pressure", m, instance))
	{
		clothSimAttribute.pressure = strtod(mj_getPluginConfig(m, instance, "pressure"), nullptr);
	}
	if (CheckNumAttr("pin", m, instance))
	{
		String2Vector(mj_getPluginConfig(m, instance, "pin"), pinVerts);
	}
	if (CheckNumAttr("solidifystiff", m, instance))
	{
		solidifyStiff = strtod(mj_getPluginConfig(m, instance, "solidifystiff"), nullptr) * 1.0e-3;
	}

	{
		const char* config = mj_getPluginConfig(m, instance, "keepwrinkles");
		if (config)
		{
			std::string tmp = config;
			keepWrinkles = tmp == "true";
		}
	}

	if (CheckNumAttr("clothfriction", m, instance))
	{
		clothSimAttribute.dynamicFriction = strtod(mj_getPluginConfig(m, instance, "clothfriction"), nullptr);
	}

	if (CheckNumAttr("clothstaticfriction", m, instance))
	{
		clothSimAttribute.staticFriction = strtod(mj_getPluginConfig(m, instance, "clothstaticfriction"), nullptr);
	}
}

Style3DSim::~Style3DSim() {}

void Style3DSim::CreateS3dMeshes(const mjModel* m, mjData* d)
{
	// use master flex to filter collision, may improve in the future
	int flex_contype = m->flex_contype[flexIdx];
	int flex_conaffinity = m->flex_conaffinity[flexIdx];

	auto& geoMeshIndexPair = Style3DSimHndManager::GetSingleton().geoMeshIndexPair;
	geoMeshIndexPair.reserve(m->ngeom);
	auto& meshHnds = Style3DSimHndManager::GetSingleton().meshHnds;
	for (int i = 0; i < m->ngeom; ++i)
	{
		int meshid = m->geom_dataid[i];
		if (meshid < 0) continue;
		if (m->geom_type[i] != mjGEOM_MESH) continue;
		if (useConvexHull && m->mesh_graphadr[meshid] < 0)
		{
			continue;
		}

		// use mujoco filter rule
		if (!(m->geom_contype[i] & flex_conaffinity) && !(m->geom_conaffinity[i] & flex_contype)) continue;

		geoMeshIndexPair.push_back(SrVec2i{ i, meshid });
		if (meshHnds.find(meshid) != meshHnds.end()) continue;

		std::vector<SrVec3f>		verts;
		std::vector<SrVec3i>		faces;

		if (useConvexHull)
		{
			int vertadr = m->mesh_vertadr[meshid];
			//int numMeshVert = m->mesh_vertnum[meshid];
			// get sizes of convex hull
			int numConvexHullVert = m->mesh_graph[m->mesh_graphadr[meshid]];
			int numConvexHullFace = m->mesh_graph[m->mesh_graphadr[meshid] + 1];
			faces.resize(numConvexHullFace);
			int* vert_globalid = m->mesh_graph + m->mesh_graphadr[meshid] + 2 + numConvexHullVert;
			int* pFaces = m->mesh_graph + m->mesh_graphadr[meshid] + 2 + 3 * numConvexHullVert + 3 * numConvexHullFace;
			float* pVerts = m->mesh_vert + 3 * vertadr;

			std::unordered_map<int, int> vertIdxMap;
			verts.reserve(numConvexHullVert);

			// rebuild vert index
			for (int i = 0; i < numConvexHullVert; ++i) {
				int vertIdx = vert_globalid[i];
				vertIdxMap[vertIdx] = i;
				verts.emplace_back(SrVec3f{ pVerts[3 * vertIdx], pVerts[3 * vertIdx + 2], -pVerts[3 * vertIdx + 1] });
			}

			for (int i = 0; i < faces.size(); i++)
			{
				faces[i] = SrVec3i{ vertIdxMap[pFaces[3 * i]], vertIdxMap[pFaces[3 * i + 1]], vertIdxMap[pFaces[3 * i + 2]] };
			}
		}
		else
		{
			int* pFaces = m->mesh_face + 3 * m->mesh_faceadr[meshid];
			float* pVerts = m->mesh_vert + 3 * m->mesh_vertadr[meshid];

			verts.resize(m->mesh_vertnum[meshid]);
			faces.resize(m->mesh_facenum[meshid]);

			for (int i = 0; i < verts.size(); i++)
			{
				verts[i] = SrVec3f{ pVerts[3 * i], pVerts[3 * i + 2], -pVerts[3 * i + 1] };
			}
			for (int i = 0; i < faces.size(); i++)
			{
				faces[i] = SrVec3i{ pFaces[3 * i], pFaces[3 * i + 1], pFaces[3 * i + 2] };
			}
		}

		SrMeshDesc meshDesc;
		meshDesc.numVertices = verts.size();
		meshDesc.numTriangles = faces.size();
		meshDesc.positions = reinterpret_cast<SrVec3f*>(verts.data());
		meshDesc.triangles = reinterpret_cast<SrVec3i*>(faces.data());
		meshHnds[meshid] = SrMesh_Create(&meshDesc);
	}
	geoMeshIndexPair.shrink_to_fit();
}

void Style3DSim::InitFlexIdx(const mjModel* m, mjData* d, int instance)
{
	int bodyId = -1;
	for (int i = 1; i < m->nbody; i++) 
	{
		if (m->body_plugin[i] == instance) 
		{
			bodyId = i;
			break;
		}
	}

	flexIdx = -1;
	for (int i = 0; i < m->nflex; i++)
	{
		int adr = m->flex_vertadr[i];
		int num = m->flex_vertnum[i];
		for (int k = 0; k < num; ++k)
		{
			if (m->flex_vertbodyid[adr + k] == bodyId)
			{
				flexIdx = i;
				return;
			}
		}
	}
}

void Style3DSim::Compute(const mjModel* m, mjData* d, int instance) {

  return;
}

void Style3DSim::Advance(const mjModel* m, mjData* d, int instance) {

	if (flexIdx < 0)
	{
		mju_error("Flex id invalid.");
		return;
	}
	int frameIndex = std::round(d->time / m->opt.timestep);
	auto& simHndManager = Style3DSimHndManager::GetSingleton();
	if (simHndManager.masterInstance == -1)
	{
		// this plugin instance is master
		simHndManager.masterInstance = instance;
	}
	bool isMaster = simHndManager.masterInstance == instance;

	int flexVertNum = m->flex_vertnum[flexIdx];
	int flexVertAdr = m->flex_vertadr[flexIdx];
	int flexElemNum = m->flex_elemnum[flexIdx];
	int flexElemAdr = m->flex_elemadr[flexIdx];
	mjtByte flexCustomFlag = m->flex_custom[flexIdx]; 

	if (frameIndex == 1) // init sim in first frame
	{
		if (isMaster)
		{
			// log callback
			auto pfnSrLogCallback = [](const char* pFileName, const char* pFunName, int line, SrLogVerb eLogVerb, const char* pMsg)
				{
					if (eLogVerb == SrLogVerb::SrInfo)
					{
						printf("Info: %s\n", pMsg);
					}
					else if (eLogVerb == SrLogVerb::SrError)
					{
						printf("Error: %s\n", pMsg);
					}
					else if (eLogVerb == SrLogVerb::SrDebug)
					{
						printf("Debug: %s\n", pMsg);
					}
					else if (eLogVerb == SrLogVerb::SrAssert)
					{
						printf("Assert: %s\n", pMsg);
					}
					else if (eLogVerb == SrLogVerb::SrWarn)
					{
						printf("Warn: %s\n", pMsg);
					}
				};
			SrSetLogCallback(pfnSrLogCallback);

			// init sim world and collider
			auto LoginCallback = [](bool bSucceed, const char* errorType, const char* message)
				{
					if (bSucceed)
					{
						printf("Succeed login.\n");
					}
					else
					{
						printf("Fail login, errorType: %s, message: %s.\n", errorType, message);
					}
				};
			if (!SrIsLogin())
				SrLogin(usr.c_str(), pwd.c_str(), true, LoginCallback);

			// create world
			simHndManager.Clear(); // clear sim hnds at first frame
			simHndManager.masterInstance = instance; // Clear func erase masterInstance, so reset again
			simHndManager.worldHnd = SrWorld_Create();
			worldSimAttribute.timeStep = m->opt.timestep / substep;
			worldSimAttribute.iterations = m->opt.iterations;
			worldSimAttribute.gravity.x = m->opt.gravity[0];
			worldSimAttribute.gravity.y = m->opt.gravity[2];
			worldSimAttribute.gravity.z = -m->opt.gravity[1];
			worldSimAttribute.enableRigidSelfCollision = false;
			SrWorld_SetAttribute(simHndManager.worldHnd, &worldSimAttribute);

			CreateS3dMeshes(m, d);

			// create collider
			if (useRigidCollider)
			{
				simHndManager.rigidHnds.resize(simHndManager.geoMeshIndexPair.size());
				for (int i = 0; i < simHndManager.geoMeshIndexPair.size(); i++)
				{
					int g = simHndManager.geoMeshIndexPair[i].x;
					int meshid = simHndManager.geoMeshIndexPair[i].y;
					const mjtNum* mat = d->geom_xmat + 9 * g;
					const mjtNum* pos = d->geom_xpos + 3 * g;
					const mjtNum geomSlideFrition = m->geom_friction[3 * g];
					// convert transform coordinate
					float rotMat[9];
					rotMat[0] = mat[0]; rotMat[1] = mat[2]; rotMat[2] = -mat[1];
					rotMat[3] = mat[6]; rotMat[4] = mat[8]; rotMat[5] = -mat[7];
					rotMat[6] = -mat[3]; rotMat[7] = -mat[5]; rotMat[8] = mat[4];
					SrTransform transform;
					transform.rotation = SrQuat_Create(rotMat);
					transform.scale = SrVec3f{ 1.0f, 1.0f, 1.0f };
					transform.translation.x = +pos[0];
					transform.translation.y = +pos[2];
					transform.translation.z = -pos[1];

					simHndManager.geoTransforms[g] = transform;
					simHndManager.geoSlideFrictions[g] = geomSlideFrition;

					simHndManager.rigidHnds[i] = SrRigidBody_Create(simHndManager.meshHnds[meshid], &transform);
					SrRigidBodySimAttribute rigidBodySimAttribute;
					rigidBodySimAttribute.staticFriction = colliderSimAttribute.staticFriction < 0 ? geomSlideFrition : colliderSimAttribute.staticFriction;
					rigidBodySimAttribute.dynamicFriction = colliderSimAttribute.dynamicFriction < 0 ? geomSlideFrition : colliderSimAttribute.dynamicFriction;
					SrRigidBody_SetAttribute(simHndManager.rigidHnds[i], &rigidBodySimAttribute);
					SrRigidBody_SetPinFlag(simHndManager.rigidHnds[i], true);
					SrRigidBody_Attach(simHndManager.rigidHnds[i], simHndManager.worldHnd);
				}
			}
			else
			{
				simHndManager.colliderHnds.resize(simHndManager.geoMeshIndexPair.size());
				for (int i = 0; i < simHndManager.geoMeshIndexPair.size(); i++)
				{
					int g = simHndManager.geoMeshIndexPair[i].x;
					int meshid = simHndManager.geoMeshIndexPair[i].y;
					const mjtNum* mat = d->geom_xmat + 9 * g;
					const mjtNum* pos = d->geom_xpos + 3 * g;
					const mjtNum geomSlideFrition = m->geom_friction[3 * g];
					// convert transform coordinate
					float rotMat[9];
					rotMat[0] = mat[0]; rotMat[1] = mat[2]; rotMat[2] = -mat[1];
					rotMat[3] = mat[6]; rotMat[4] = mat[8]; rotMat[5] = -mat[7];
					rotMat[6] = -mat[3]; rotMat[7] = -mat[5]; rotMat[8] = mat[4];
					SrTransform transform;
					transform.rotation = SrQuat_Create(rotMat);
					transform.scale = SrVec3f{ 1.0f, 1.0f, 1.0f };
					transform.translation.x = +pos[0];
					transform.translation.y = +pos[2];
					transform.translation.z = -pos[1];

					simHndManager.geoTransforms[g] = transform;
					simHndManager.geoSlideFrictions[g] = geomSlideFrition;

					const SrVec3f* verts = SrMesh_GetVertPositions(simHndManager.meshHnds[meshid]);
					size_t vertNum = SrMesh_GetVertNumber(simHndManager.meshHnds[meshid]);
					const SrVec3i* faces = SrMesh_GetTriangles(simHndManager.meshHnds[meshid]);
					size_t faceNum = SrMesh_GetTriangleNumber(simHndManager.meshHnds[meshid]);

					std::vector<SrVec3f> postions(vertNum);
					for (size_t vId = 0; vId < postions.size(); vId++)
					{
						postions[vId] = SrTransform_TransformVec3f(&transform, &verts[vId]);
					}
					std::vector<SrVec3i> triangles(faceNum);
					for (size_t tId = 0; tId < faceNum; tId++)
					{
						triangles[tId] = faces[tId];
					}

					// create mesh collider
					SrMeshDesc colliderMeshDesc;
					colliderMeshDesc.numVertices = postions.size();
					colliderMeshDesc.numTriangles = faceNum;
					colliderMeshDesc.positions = postions.data();
					colliderMeshDesc.triangles = triangles.data();
					simHndManager.colliderHnds[i] = SrMeshCollider_Create(&colliderMeshDesc);
					auto curColliderSimAttribute = colliderSimAttribute;
					curColliderSimAttribute.staticFriction = colliderSimAttribute.staticFriction < 0 ? geomSlideFrition : colliderSimAttribute.staticFriction;
					curColliderSimAttribute.dynamicFriction = colliderSimAttribute.dynamicFriction < 0 ? geomSlideFrition : colliderSimAttribute.dynamicFriction;
					SrMeshCollider_SetAttribute(simHndManager.colliderHnds[i], &curColliderSimAttribute);
					SrMeshCollider_Attach(simHndManager.colliderHnds[i], simHndManager.worldHnd);
				}
			}
		}

		// create cloth verts
		std::vector<SrVec3f>	pos(flexVertNum);
		std::vector<SrVec2f>	materialCoords(flexVertNum);
		std::vector<char>	isPinned(pinVerts.size(), 1);
		for (int i = 0; i < flexVertNum; ++i)
		{
			int vid = flexVertAdr + i;
			pos[i].x = d->flexvert_xpos[3 * vid + 0];
			pos[i].y = d->flexvert_xpos[3 * vid + 2];
			pos[i].z = -d->flexvert_xpos[3 * vid + 1];

			materialCoords[i].x = pos[i].x;
			materialCoords[i].y = pos[i].z;
		}

		// create cloth faces, we only consider cloth type for now
		std::vector<SrVec3i> clothFaces(flexElemNum);
		for (int i = 0; i < flexElemNum; ++i)
		{
			int elemid = flexElemAdr + i;
			clothFaces[i].x = m->flex_elem[3 * elemid];
			clothFaces[i].y = m->flex_elem[3 * elemid + 1];
			clothFaces[i].z = m->flex_elem[3 * elemid + 2];
		}

		//create cloth
		SrMeshDesc meshDesc;
		meshDesc.numVertices = pos.size() -1 ; // trick, last vert is ghost
		meshDesc.numTriangles = clothFaces.size();
		meshDesc.positions = pos.data();
		meshDesc.triangles = clothFaces.data();
		simHndManager.clothHnds[flexIdx] = SrCloth_Create(&meshDesc, nullptr, keepWrinkles);
		//if (clothSimAttribute.staticFriction < clothSimAttribute.dynamicFriction)
		//	clothSimAttribute.staticFriction = clothSimAttribute.dynamicFriction;
		SrCloth_SetAttribute(simHndManager.clothHnds[flexIdx], &clothSimAttribute);

		if (pinVerts.size() > 0)
		{
			SrCloth_SetVertPinFlags(simHndManager.clothHnds[flexIdx], pinVerts.size(), (bool*)isPinned.data(), pinVerts.data());
		}

		if (solidifyStiff > 1.0e-3)
		{
			std::vector<float>	solidifyStiffs(m->nflexvert);
			std::vector<int>	solidifyVerts(m->nflexvert);
			for (int i = 0; i < m->nflexvert; i++)
			{
				solidifyStiffs[i] = solidifyStiff;
				solidifyVerts[i] = i;
			}
			SrCloth_Solidify(simHndManager.clothHnds[flexIdx], simHndManager.worldHnd, meshDesc.numVertices, solidifyStiffs.data(), solidifyVerts.data());
		}

		SrCloth_Attach(simHndManager.clothHnds[flexIdx], simHndManager.worldHnd);
	}
	else if (frameIndex > 1)
	{
		if (isMaster)
		{
			// update collider transform by master
			auto& geoMeshIndexPair = simHndManager.geoMeshIndexPair;
			if (useRigidCollider)
			{
				for (int i = 0; i < simHndManager.geoMeshIndexPair.size(); i++)
				{
					int g = geoMeshIndexPair[i].x;
					const mjtNum* mat = d->geom_xmat + 9 * g;
					const mjtNum* pos = d->geom_xpos + 3 * g;
					const mjtNum geomSlideFrition = m->geom_friction[3 * g];
					if (!IsEqual(simHndManager.geoSlideFrictions[g], geomSlideFrition))
					{
						SrRigidBodySimAttribute rigidBodySimAttribute;
						rigidBodySimAttribute.staticFriction = colliderSimAttribute.staticFriction < 0 ? geomSlideFrition : colliderSimAttribute.staticFriction;
						rigidBodySimAttribute.dynamicFriction = colliderSimAttribute.dynamicFriction < 0 ? geomSlideFrition : colliderSimAttribute.dynamicFriction;
						SrRigidBody_SetAttribute(simHndManager.rigidHnds[i], &rigidBodySimAttribute);
						simHndManager.geoSlideFrictions[g] = geomSlideFrition;
					}
					// convert transform coordinate
					float rotMat[9];
					rotMat[0] = mat[0]; rotMat[1] = mat[2]; rotMat[2] = -mat[1];
					rotMat[3] = mat[6]; rotMat[4] = mat[8]; rotMat[5] = -mat[7];
					rotMat[6] = -mat[3]; rotMat[7] = -mat[5]; rotMat[8] = mat[4];
					SrTransform transform;
					transform.rotation = SrQuat_Create(rotMat);
					transform.scale = SrVec3f{ 1.0f, 1.0f, 1.0f };
					transform.translation.x = +pos[0];
					transform.translation.y = +pos[2];
					transform.translation.z = -pos[1];

					if (!IsEqual(simHndManager.geoTransforms[g], transform))
					{
						SrRigidBody_Move(simHndManager.rigidHnds[i], &simHndManager.geoTransforms[g], &transform);
						simHndManager.geoTransforms[g] = transform;
					}
				}
			}
			else
			{
				for (int i = 0; i < geoMeshIndexPair.size(); i++)
				{
					int g = geoMeshIndexPair[i].x;
					int meshid = geoMeshIndexPair[i].y;
					const mjtNum* mat = d->geom_xmat + 9 * g;
					const mjtNum* pos = d->geom_xpos + 3 * g;
					const mjtNum geomSlideFrition = m->geom_friction[3 * g];
					if (!IsEqual(simHndManager.geoSlideFrictions[g], geomSlideFrition))
					{
						auto curColliderSimAttribute = colliderSimAttribute;
						curColliderSimAttribute.staticFriction = colliderSimAttribute.staticFriction < 0 ? geomSlideFrition : colliderSimAttribute.staticFriction;
						curColliderSimAttribute.dynamicFriction = colliderSimAttribute.dynamicFriction < 0 ? geomSlideFrition : colliderSimAttribute.dynamicFriction;
						SrMeshCollider_SetAttribute(simHndManager.colliderHnds[i], &curColliderSimAttribute);
						simHndManager.geoSlideFrictions[g] = geomSlideFrition;
					}
					// convert transform coordinate
					float rotMat[9];
					rotMat[0] = mat[0]; rotMat[1] = mat[2]; rotMat[2] = -mat[1];
					rotMat[3] = mat[6]; rotMat[4] = mat[8]; rotMat[5] = -mat[7];
					rotMat[6] = -mat[3]; rotMat[7] = -mat[5]; rotMat[8] = mat[4];
					SrTransform transform;
					transform.rotation = SrQuat_Create(rotMat);
					transform.scale = SrVec3f{ 1.0f, 1.0f, 1.0f };
					transform.translation.x = +pos[0];
					transform.translation.y = +pos[2];
					transform.translation.z = -pos[1];

					if (!IsEqual(simHndManager.geoTransforms[g], transform))
					{
						simHndManager.geoTransforms[g] = transform;

						const SrVec3f* verts = SrMesh_GetVertPositions(simHndManager.meshHnds[meshid]);
						size_t vertNum = SrMesh_GetVertNumber(simHndManager.meshHnds[meshid]);
						//const SrVec3i* faces = SrMesh_GetTriangles(simHndManager->meshHnds[meshid]);
						//size_t faceNum = SrMesh_GetTriangleNumber(simHndManager->meshHnds[meshid]);

						std::vector<SrVec3f> postions(vertNum);
						for (size_t vId = 0; vId < postions.size(); vId++)
						{
							postions[vId] = SrTransform_TransformVec3f(&transform, &verts[vId]);
						}

						SrMeshCollider_MoveVerts(simHndManager.colliderHnds[i], postions.size(), nullptr, postions.data());
					}
				}
			}
		}

		if (flexCustomFlag & (mjtByte)EStyle3DCustomBit::ClearPin)
		{
			if (pinVerts.size() > 0)
			{
				std::vector<char> isPinned(pinVerts.size(), 0);
				SrCloth_SetVertPinFlags(simHndManager.clothHnds[flexIdx], pinVerts.size(), (bool*)isPinned.data(), pinVerts.data());
			}
		}
		else if (flexCustomFlag & (mjtByte)EStyle3DCustomBit::RestorePin)
		{
			if (pinVerts.size() > 0)
			{
				std::vector<char> isPinned(pinVerts.size(), 1);
				SrCloth_SetVertPinFlags(simHndManager.clothHnds[flexIdx], pinVerts.size(), (bool*)isPinned.data(), pinVerts.data());
			}
		}

		if (flexCustomFlag & (mjtByte)EStyle3DCustomBit::SetPos)
		{
			std::vector<SrVec3f> pos(flexVertNum);
			std::vector<int> vertIndices(flexVertNum);
			for (int i = 0; i < flexVertNum; ++i)
			{
				int vid = flexVertAdr + i;
				pos[i].x = d->flexvert_xpos[3 * vid + 0];
				pos[i].y = d->flexvert_xpos[3 * vid + 2];
				pos[i].z = -d->flexvert_xpos[3 * vid + 1];
				vertIndices[i] = i;
			}
			SrCloth_SetVertPositions(simHndManager.clothHnds[flexIdx], flexVertNum - 1, pos.data(), vertIndices.data());
		}
		else
		{
			// set cloth positions for pin verts
			int numPin = pinVerts.size();
			if (numPin > 0)
			{
				std::vector<SrVec3f> pos(numPin);
				for (int i = 0; i < numPin; i++)
				{
					int v = pinVerts[i] + flexVertAdr;
					pos[i].x = d->flexvert_xpos[3 * v + 0];
					pos[i].y = d->flexvert_xpos[3 * v + 2];
					pos[i].z = -d->flexvert_xpos[3 * v + 1];
				}

				SrCloth_SetVertPositions(simHndManager.clothHnds[flexIdx], numPin, pos.data(), pinVerts.data());
			}
		}

		if (flexCustomFlag & (mjtByte)EStyle3DCustomBit::ClearVel)
		{
			std::vector<SrVec3f> vel(flexVertNum);
			std::vector<int> vertIndices(flexVertNum);
			for (int i = 0; i < flexVertNum; ++i)
			{
				vel[i].x = vel[i].y = vel[i].z = 0.0f;
				vertIndices[i] = i;
			}
			SrCloth_SetVertVelocities(simHndManager.clothHnds[flexIdx], flexVertNum - 1, vel.data(), vertIndices.data());
		}
	}

	if (isMaster)
	{
		if (!(flexCustomFlag & (mjtByte)EStyle3DCustomBit::SkipSim))
		{
			for (int s = 0; s < substep; ++s)
				SrWorld_StepSim(simHndManager.worldHnd);
			SrWorld_FetchSim(simHndManager.worldHnd);
		}
		frameIndex++;
	}

	if (frameIndex > 3) // trick, skip  first 3 frame, because mj do step when compile model
	{
		const SrVec3f* pos = SrCloth_GetVertPositions(simHndManager.clothHnds[flexIdx]);
		for (int i = 0; i < flexVertNum - 1; ++i)// trick, last vert is ghost
		{
			int vid = flexVertAdr + i;
			d->flexvert_xpos[3 * vid + 0] = pos[i].x;
			d->flexvert_xpos[3 * vid + 2] = pos[i].y;
			d->flexvert_xpos[3 * vid + 1] = -pos[i].z;
		}

		// reset the ghost vert for numeric safety
		int ghostVertIdx = flexVertAdr + flexVertNum - 1;
		mju_zero3(d->xpos + 3 * m->flex_vertbodyid[ghostVertIdx]);
	}
}

void Style3DSim::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.style3dsim.style3dsim";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;

  const char* attributes[] = {"face", "edge", 
							  "stretch", "bend", "thickness", "density", "pressure", "solidifystiff","pin",
							  "staticfriction", "friction", "groundstaticfriction", "groundfriction", "clothstaticfriction", "clothfriction", 
							  "gap", "convex", "rigidcollider", "selfcollide", "keepwrinkles",
							  "airdamping", "stretchdamping", "benddamping", "velsmoothing", "groundheight", "gpu", "substep",
							  "user", "pwd"};
  plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
  plugin.attributes = attributes;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto style3d_or_null = Style3DSim::Create(m, d, instance);
    if (!style3d_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new Style3DSim(std::move(*style3d_or_null)));
return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Style3DSim*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;

	auto& masterInstance = Style3DSimHndManager::GetSingleton().masterInstance;
	if (masterInstance == instance)
	{
		// this plugin instance is master, clear manager
		Style3DSimHndManager::GetSingleton().Clear();
	}

  };
   plugin.compute =
       +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
         auto* style3dsim = reinterpret_cast<Style3DSim*>(d->plugin_data[instance]);
         style3dsim->Compute(m, d, instance);
       };
   plugin.advance =
       +[](const mjModel* m, mjData* d, int instance) {
       auto* style3dsim = reinterpret_cast<Style3DSim*>(d->plugin_data[instance]);
       style3dsim->Advance(m, d, instance);
       };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::style3dsim
