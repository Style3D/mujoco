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

#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include "style3dsim.h"

namespace mujoco::plugin::style3dsim {
namespace {

// reads numeric attributes
bool CheckAttr(const char* name, const mjModel* m, int instance) {
  char *end;
  std::string value = mj_getPluginConfig(m, instance, name);
  value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
  strtod(value.c_str(), &end);
  return end == value.data() + value.size();
}

void String2Vector(const std::string& txt, std::vector<int>& vec) {
	std::stringstream strm(txt);
	vec.clear();

	while (!strm.eof()) {
		int num;
		strm >> num;
		if (strm.fail()) {
			break;
		}
		else {
			vec.push_back(num);
		}
	}
}

}  // namespace

Style3DSimHndManager::~Style3DSimHndManager() {

	SrWorld_Destroy(&worldHnd);
	SrCloth_Destroy(&clothHnd);
	for (auto& hnd : colliderHnds)
	{
		SrMeshCollider_Destroy(&hnd);
	}
	colliderHnds.clear();
	for (auto& hnd : meshHnds)
	{
		SrMesh_Destroy(&hnd.second);
	}
	meshHnds.clear();
}

// factory function
std::optional<Style3DSim> Style3DSim::Create(
  const mjModel* m, mjData* d, int instance) {

	if (CheckAttr("face", m, instance))
	{
		std::vector<int> face;
		String2Vector(mj_getPluginConfig(m, instance, "face"), face);
		return Style3DSim(m, d, instance, face);
	}
	else
	{
		mju_warning("Invalid parameter specification in shell plugin");
		return std::nullopt;
	}
}

void Style3DSim::CreateStaticMeshes(const mjModel* m, mjData* d)
{
	geoMeshIndexPair.reserve(m->ngeom);
	for (int i = 0; i < m->ngeom; ++i)
	{
		int meshid = m->geom_dataid[i];
		if (meshid < 0) continue;

		if (useConvexHull)
		{
			if (m->geom_type[i] != mjGEOM_MESH) continue;
			if (m->mesh_graphadr[meshid] < 0) continue;
			if (!m->geom_contype[i] && !m->geom_conaffinity[i]) continue;
		}

		geoMeshIndexPair.push_back(SrVec2i{ i, meshid });
		if (simHndManager->meshHnds.find(meshid) != simHndManager->meshHnds.end()) continue;

		std::vector<SrVec3f>		verts;
		std::vector<SrVec3i>		faces;

		if (useConvexHull)
		{
			int vertadr = m->mesh_vertadr[meshid];
			int numMeshVert = m->mesh_vertnum[meshid];
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
				verts.emplace_back(SrVec3f{pVerts[3 * vertIdx], pVerts[3 * vertIdx + 2], -pVerts[3 * vertIdx + 1]});
			}

			for (int i = 0; i < faces.size(); i++)
			{
				faces[i] = SrVec3i{vertIdxMap[pFaces[3 * i]], vertIdxMap[pFaces[3 * i + 1]], vertIdxMap[pFaces[3 * i + 2]]};
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
				verts[i] = SrVec3f{pVerts[3 * i], pVerts[3 * i + 2], -pVerts[3 * i + 1]};
			}
			for (int i = 0; i < faces.size(); i++)
			{
				faces[i] = SrVec3i{pFaces[3 * i], pFaces[3 * i + 1], pFaces[3 * i + 2]};
			}
		}

		SrMeshDesc meshDesc;
		meshDesc.numVertices = verts.size();
		meshDesc.numTriangles = faces.size();
		meshDesc.positions = reinterpret_cast<SrVec3f*>(verts.data());
		meshDesc.triangles = reinterpret_cast<SrVec3i*>(faces.data());
		simHndManager->meshHnds[meshid] = SrMesh_Create(&meshDesc);
	}
	geoMeshIndexPair.shrink_to_fit();
}

// plugin constructor
Style3DSim::Style3DSim(const mjModel* m, mjData* d, int instance, const std::vector<int>& face) {

  mjtNum Stretch = strtod(mj_getPluginConfig(m, instance, "stretch"), nullptr);
  //printf("Style3D Stretch %lf", Stretch);

  clothFaces.resize(face.size() / 3);

  std::memcpy(clothFaces.data(), face.data(), sizeof(int) * face.size());

  simHndManager = std::make_shared<Style3DSimHndManager>();
}

Style3DSim::~Style3DSim() {
}

void Style3DSim::Compute(const mjModel* m, mjData* d, int instance) {

  return;
}

void Style3DSim::Advance(const mjModel* m, mjData* d, int instance) {

	if (frameIndex == 0)
	{
		// init sim
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
		SrLogin("user", "pwd", true, LoginCallback);

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

		simHndManager->worldHnd = SrWorld_Create();
		SrWorldSimAttribute worldSimAttribute;
		worldSimAttribute.enableGPU = false;
		worldSimAttribute.timeStep = m->opt.timestep;
		SrWorld_SetAttribute(simHndManager->worldHnd, &worldSimAttribute);

		std::vector<SrVec3f>	pos(m->nflexvert);
		std::vector<SrVec2f>	materialCoords(m->nflexvert);
		std::vector<bool>	isPinned(m->nflexvert, false);

		for (int i = 0; i < pos.size(); i++)
		{
			pos[i].x = d->flexvert_xpos[3 * i + 0];
			pos[i].y = d->flexvert_xpos[3 * i + 2];
			pos[i].z = -d->flexvert_xpos[3 * i + 1];

			materialCoords[i].x = pos[i].x;
			materialCoords[i].y = pos[i].z;
		}

		//create cloth
		SrMeshDesc meshDesc;
		meshDesc.numVertices = pos.size() -1 ; // trick, last vert is ghost
		meshDesc.numTriangles = clothFaces.size();
		meshDesc.positions = pos.data();
		meshDesc.triangles = clothFaces.data();
		simHndManager->clothHnd = SrCloth_Create(&meshDesc);
		SrClothSimAttribute clothSimAttribute;
		SrCloth_SetAttribute(simHndManager->clothHnd, &clothSimAttribute);
		SrCloth_Attach(simHndManager->clothHnd, simHndManager->worldHnd);

		CreateStaticMeshes(m, d);

		// create collider
		simHndManager->colliderHnds.resize(geoMeshIndexPair.size());
		for (int i = 0; i < geoMeshIndexPair.size(); i++)
		{
			int g = geoMeshIndexPair[i].x;
			int meshid = geoMeshIndexPair[i].y;
			const mjtNum* mat = d->geom_xmat + 9 * g;
			const mjtNum* pos = d->geom_xpos + 3 * g;
			// convert transform coordinate
			float rotMat[9]; 
			rotMat[0] = mat[0]; rotMat[1] = mat[2]; rotMat[2] = -mat[1];
			rotMat[3] = mat[6]; rotMat[4] = mat[8]; rotMat[5] = -mat[7];
			rotMat[6] = -mat[3]; rotMat[7] = -mat[5]; rotMat[8] = mat[4];
			SrTransform transform;
			transform.rotation = SrQuat_Create(rotMat);
			transform.scale = SrVec3f{1.0f, 1.0f, 1.0f};
			transform.translation.x = +pos[0];
			transform.translation.y = +pos[2];
			transform.translation.z = -pos[1];

			const SrVec3f* verts = SrMesh_GetVertPositions(simHndManager->meshHnds[meshid]);
			size_t vertNum = SrMesh_GetVertNumber(simHndManager->meshHnds[meshid]);
			const SrVec3i* faces = SrMesh_GetTriangles(simHndManager->meshHnds[meshid]);
			size_t faceNum = SrMesh_GetTriangleNumber(simHndManager->meshHnds[meshid]);

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
			simHndManager->colliderHnds[i] = SrMeshCollider_Create(&colliderMeshDesc);
			SrColliderSimAttribute colliderSimAttribute;
			SrMeshCollider_SetAttribute(simHndManager->colliderHnds[i], &colliderSimAttribute);
			SrMeshCollider_Attach(simHndManager->colliderHnds[i], simHndManager->worldHnd);
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

			const SrVec3f* verts = SrMesh_GetVertPositions(simHndManager->meshHnds[meshid]);
			size_t vertNum = SrMesh_GetVertNumber(simHndManager->meshHnds[meshid]);
			const SrVec3i* faces = SrMesh_GetTriangles(simHndManager->meshHnds[meshid]);
			size_t faceNum = SrMesh_GetTriangleNumber(simHndManager->meshHnds[meshid]);

			std::vector<SrVec3f> postions(vertNum);
			for (size_t vId = 0; vId < postions.size(); vId++)
			{
				postions[vId] = SrTransform_TransformVec3f(&transform, &verts[vId]);
			}

			SrMeshCollider_SetVertPositions(simHndManager->colliderHnds[i], postions.size(), nullptr, postions.data());
		}
	}

	SrWorld_StepSim(simHndManager->worldHnd);
	bool isCaptured = SrWorld_FetchSim(simHndManager->worldHnd);
	if (isCaptured)
	{
		const SrVec3f* pos = SrCloth_GetVertPositions(simHndManager->clothHnd);
		for (int i = 0; i < m->nflexvert - 1; i++)// trick, last vert is ghost
		{
			m->flex_vert[3 * i + 0] = pos[i].x;
			m->flex_vert[3 * i + 2] = pos[i].y;
			m->flex_vert[3 * i + 1] = -pos[i].z;
		}
	}

	frameIndex++;
}

void Style3DSim::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.style3dsim.style3dsim";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;

  const char* attributes[] = {"face", "edge", "thickness", "damping", "stretch", "bend"};
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
