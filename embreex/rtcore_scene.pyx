# distutils: language=c++
# Embree scene.

cimport cython
cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from numpy cimport int32_t
import numpy as np
import logging
import numbers
from . cimport rtcore as rtc
from . cimport rtcore_buffer as rtcb
from . cimport rtcore_ray as rtcr
from . cimport rtcore_scene as rtcs
from . cimport rtcore_geometry as rtcg
from .rtcore cimport Vertex, Triangle

log = logging.getLogger(__name__)

cdef void error_printer(void *userPtr, rtc.RTCError code, const char *_str) noexcept:
    """
    Función callback para errores en Embree.
    """
    log.error("ERROR CAUGHT IN EMBREE")
    rtc.print_error(code)
    log.error("ERROR MESSAGE: %s" % _str)

cdef packed struct hit_struct:
    np.int32_t geomID
    np.int32_t primID
    np.int32_t rayIDX
    np.float32_t tfar

cdef packed struct hit_count_struct:
    np.int32_t primID
    np.int32_t count
    np.float32_t weight

cdef class EmbreeScene:
    def __init__(self, rtc.EmbreeDevice device=None):
        if device is None:
            device = rtc.EmbreeDevice()
        self.device = device
        rtc.rtcSetDeviceErrorFunction(device.device, error_printer, NULL)
        # Usar la función de escena del módulo rtcs
        self.scene_i = rtcs.rtcNewScene(device.device)
        self.is_committed = 0

    def set_build_quality(self, rtc.RTCBuildQuality quality):
        rtcs.rtcSetSceneBuildQuality(self.scene_i, quality)

    def get_flags(self):
        return rtcs.rtcGetSceneFlags(self.scene_i)

    def set_flags(self, int flags):
        rtcs.rtcSetSceneFlags(self.scene_i, flags)

    def commit(self):
        rtcs.rtcCommitScene(self.scene_i)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def run(self, np.ndarray[np.float32_t, ndim=2] vec_origins,
                  np.ndarray[np.float32_t, ndim=2] vec_directions,
                  dists=None, query='INTERSECT', output=None):

        if self.is_committed == 0:
            rtcs.rtcCommitScene(self.scene_i)
            self.is_committed = 1

        cdef rtcs.RTCBounds bnds
        rtcs.rtcGetSceneBounds(self.scene_i, &bnds)

        cdef int nv = vec_origins.shape[0]
        cdef int vd_i, vd_step
        cdef np.ndarray[np.int32_t, ndim=1] intersect_ids
        cdef np.ndarray[np.int32_t, ndim=1] primID_arr
        cdef np.ndarray[np.int32_t, ndim=1] geomID_arr
        cdef np.ndarray[np.float32_t, ndim=1] tfars
        cdef np.ndarray[np.float32_t, ndim=1] u_arr
        cdef np.ndarray[np.float32_t, ndim=1] v_arr
        cdef np.ndarray[np.float32_t, ndim=2] Ng
        cdef int32_t INVALID_GEOMETRY_ID = rtcg.RTC_INVALID_GEOMETRY_ID

        cdef int query_type
        if query == 'INTERSECT':
            query_type = rtcs.intersect
        elif query == 'OCCLUDED':
            query_type = rtcs.occluded
        elif query == 'DISTANCE':
            query_type = rtcs.distance
        else:
            raise ValueError("Embree ray query type %s not recognized. Accepted types: INTERSECT, OCCLUDED, DISTANCE" % query)

        if dists is None:
            tfars = np.empty(nv, dtype="float32")
            tfars.fill(1e37)
        elif isinstance(dists, numbers.Number):
            tfars = np.empty(nv, dtype="float32")
            tfars.fill(dists)
        else:
            tfars = dists

        if output:
            u_arr = np.empty(nv, dtype="float32")
            v_arr = np.empty(nv, dtype="float32")
            Ng = np.empty((nv, 3), dtype="float32")
            primID_arr = np.empty(nv, dtype="int32")
            geomID_arr = np.empty(nv, dtype="int32")
        else:
            intersect_ids = np.empty(nv, dtype="int32")
            intersect_ids.fill(INVALID_GEOMETRY_ID)

        cdef rtcr.RTCIntersectContext context
        rtcr.rtcInitIntersectContext(&context)

        cdef rtcr.RTCRayHit ray_hit
        vd_i = 0
        vd_step = 1
        if vec_directions.shape[0] == 1:
            vd_step = 0

        cdef rtcr.RTCIntersectArguments intersect_args
        cdef rtcr.RTCOccludedArguments occluded_args
        rtcr.rtcInitIntersectArguments(&intersect_args)
        rtcr.rtcInitOccludedArguments(&occluded_args)

        for i in range(nv):
            ray_hit.ray.org_x = vec_origins[i, 0]
            ray_hit.ray.org_y = vec_origins[i, 1]
            ray_hit.ray.org_z = vec_origins[i, 2]
            ray_hit.ray.dir_x = vec_directions[vd_i, 0]
            ray_hit.ray.dir_y = vec_directions[vd_i, 1]
            ray_hit.ray.dir_z = vec_directions[vd_i, 2]
            ray_hit.ray.time = 0
            ray_hit.ray.mask = -1
            ray_hit.ray.flags = 0
            ray_hit.ray.tnear = 0.0
            ray_hit.ray.tfar = tfars[i]
            ray_hit.ray.id = i
            ray_hit.hit.geomID = rtcg.RTC_INVALID_GEOMETRY_ID
            ray_hit.hit.primID = rtcg.RTC_INVALID_GEOMETRY_ID

            vd_i += vd_step

            if query_type == rtcs.intersect or query_type == rtcs.distance:
                rtcs.rtcIntersect1(self.scene_i, &context, &ray_hit, &intersect_args)
                if not output:
                    if query_type == rtcs.intersect:
                        intersect_ids[i] = ray_hit.hit.primID
                    else:
                        tfars[i] = ray_hit.ray.tfar
                else:
                    primID_arr[i] = <int32_t>ray_hit.hit.primID
                    geomID_arr[i] = <int32_t>ray_hit.hit.geomID
                    u_arr[i] = ray_hit.hit.u
                    v_arr[i] = ray_hit.hit.v
                    tfars[i] = ray_hit.ray.tfar
                    Ng[i, 0] = ray_hit.hit.Ng_x
                    Ng[i, 1] = ray_hit.hit.Ng_y
                    Ng[i, 2] = ray_hit.hit.Ng_z
            else:
                rtcs.rtcOccluded1(self.scene_i, &context, &ray_hit.ray, &occluded_args)
                intersect_ids[i] = <int32_t>ray_hit.hit.geomID

        if output:
            return {'u': u_arr, 'v': v_arr, 'Ng': Ng, 'tfar': tfars, 'primID': primID_arr, 'geomID': geomID_arr}
        else:
            if query_type == rtcs.distance:
                return tfars
            else:
                return intersect_ids

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def multi_hit_intersect_first_gid(self, np.ndarray[np.float32_t, ndim=2] vec_origins,
                                      np.ndarray[np.float32_t, ndim=2] vec_directions, float eps=1.0e-5):
        if self.is_committed == 0:
            rtcs.rtcCommitScene(self.scene_i)
            self.is_committed = 1

        cdef int nv = vec_origins.shape[0]
        dtyp = [("geomID", np.int32), ("primID", np.int32), ("rayIDX", np.int32), ("tfar", np.float32)]
        cdef vector[hit_struct] hit_stlvec
        cdef hit_struct hit
        hit_stlvec.reserve(max(16, nv))

        cdef rtcr.RTCIntersectContext context
        rtcr.rtcInitIntersectContext(&context)
        cdef rtcr.RTCIntersectArguments intersect_args
        rtcr.rtcInitIntersectArguments(&intersect_args)
        cdef rtcr.RTCRayHit ray_hit
        cdef bool have_hit
        cdef float tnear
        cdef unordered_set[int32_t] gid_set
        for i in range(nv):
            have_hit = True
            tnear = 0.0
            gid_set.clear()
            while have_hit:
                ray_hit.ray.org_x = vec_origins[i, 0]
                ray_hit.ray.org_y = vec_origins[i, 1]
                ray_hit.ray.org_z = vec_origins[i, 2]
                ray_hit.ray.dir_x = vec_origins[i, 0]  # Asumiendo misma dirección; ajustar según sea necesario.
                ray_hit.ray.dir_y = vec_origins[i, 1]
                ray_hit.ray.dir_z = vec_origins[i, 2]
                ray_hit.ray.time = 0
                ray_hit.ray.mask = -1
                ray_hit.ray.flags = 0
                ray_hit.ray.tnear = tnear
                ray_hit.ray.tfar = np.inf
                ray_hit.ray.id = i
                ray_hit.hit.geomID = rtcg.RTC_INVALID_GEOMETRY_ID
                ray_hit.hit.primID = rtcg.RTC_INVALID_GEOMETRY_ID

                rtcs.rtcIntersect1(self.scene_i, &context, &ray_hit, &intersect_args)
                if ray_hit.hit.geomID != rtcg.RTC_INVALID_GEOMETRY_ID:
                    if gid_set.find(ray_hit.hit.geomID) == gid_set.end():
                        gid_set.insert(ray_hit.hit.geomID)
                        hit.geomID = <int32_t>ray_hit.hit.geomID
                        hit.primID = <int32_t>ray_hit.hit.primID
                        hit.rayIDX = <int32_t>i
                        hit.tfar = ray_hit.ray.tfar
                        hit_stlvec.push_back(hit)
                    tnear = ray_hit.ray.tfar + eps
                else:
                    have_hit = False

        cdef bint have_any_hits = hit_stlvec.size() > 0
        if not have_any_hits:
            hit_stlvec.push_back(hit)
        cdef hit_struct[::1] arr = <hit_struct[:] > hit_stlvec.data()
        if have_any_hits:
            ret_ary = np.asarray(arr).copy()
        else:
            ret_ary = np.empty((0,), dtype=dtyp)
        return ret_ary

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def first_hit_intersect_pid_count_with_weight(self, np.ndarray[np.float32_t, ndim=2] vec_origins,
                                                   np.ndarray[np.float32_t, ndim=2] vec_directions,
                                                   np.ndarray[np.float32_t, ndim=1] vec_weights):
        if self.is_committed == 0:
            rtcs.rtcCommitScene(self.scene_i)
            self.is_committed = 1

        cdef int nv = vec_origins.shape[0]
        cdef rtcr.RTCIntersectContext context
        rtcr.rtcInitIntersectContext(&context)
        cdef rtcr.RTCIntersectArguments intersect_args
        rtcr.rtcInitIntersectArguments(&intersect_args)
        cdef rtcr.RTCRayHit ray_hit
        cdef float tnear
        cdef unordered_map[int32_t, unordered_map[int32_t, hit_count_struct]] hit_counts_map
        for i in range(nv):
            ray_hit.ray.org_x = vec_origins[i, 0]
            ray_hit.ray.org_y = vec_origins[i, 1]
            ray_hit.ray.org_z = vec_origins[i, 2]
            ray_hit.ray.dir_x = vec_directions[i, 0]
            ray_hit.ray.dir_y = vec_directions[i, 1]
            ray_hit.ray.dir_z = vec_directions[i, 2]
            ray_hit.ray.time = 0
            ray_hit.ray.mask = -1
            ray_hit.ray.flags = 0
            ray_hit.ray.tnear = 0.0
            ray_hit.ray.tfar = np.inf
            ray_hit.ray.id = i
            ray_hit.hit.geomID = rtcg.RTC_INVALID_GEOMETRY_ID
            ray_hit.hit.primID = rtcg.RTC_INVALID_GEOMETRY_ID

            rtcs.rtcIntersect1(self.scene_i, &context, &ray_hit, &intersect_args)

            if ray_hit.hit.geomID != rtcg.RTC_INVALID_GEOMETRY_ID:
                if hit_counts_map.find(ray_hit.hit.geomID) == hit_counts_map.end():
                    hit_counts_map[ray_hit.hit.geomID] = unordered_map[int32_t, hit_count_struct]()
                if hit_counts_map[ray_hit.hit.geomID].find(ray_hit.hit.primID) == hit_counts_map[ray_hit.hit.geomID].end():
                    hit_counts_map[ray_hit.hit.geomID][ray_hit.hit.primID].primID = ray_hit.hit.primID
                    hit_counts_map[ray_hit.hit.geomID][ray_hit.hit.primID].count = 0
                    hit_counts_map[ray_hit.hit.geomID][ray_hit.hit.primID].weight = 0.0
                hit_counts_map[ray_hit.hit.geomID][ray_hit.hit.primID].count += 1
                hit_counts_map[ray_hit.hit.geomID][ray_hit.hit.primID].weight += vec_weights[i]

        cdef dict ret_dict = {}
        cdef vector[hit_count_struct] hit_count_stlvec
        cdef hit_count_struct[::1] hit_count_arr
        for gid_it in hit_counts_map:
            hit_count_stlvec.clear()
            for pid_it in hit_counts_map[gid_it.first]:
                hit_count_stlvec.push_back(pid_it.second)
            hit_count_arr = <hit_count_struct[:] > hit_count_stlvec.data()
            ret_dict[gid_it.first] = np.asarray(hit_count_arr).copy()
        return ret_dict

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def first_hit_intersect_pid_count(self, np.ndarray[np.float32_t, ndim=2] vec_origins,
                                      np.ndarray[np.float32_t, ndim=2] vec_directions):
        cdef np.ndarray[np.float32_t, ndim=1] vec_weights = np.zeros(vec_directions.shape[0], dtype="float32")
        cdef dict w_dict = self.first_hit_intersect_pid_count_with_weight(vec_origins, vec_directions, vec_weights)
        cdef dict ret_dict = {}
        for item in w_dict.items():
            ret_dict[item[0]] = np.asarray([item[1]["primID"], item[1]["count"]]).T.copy()
        return ret_dict

    def __dealloc__(self):
        rtcs.rtcReleaseScene(self.scene_i)

cdef class EmbreeSceneExtended:
    cdef rtcs.RTCScene scene_i
    cdef rtc.EmbreeDevice device
    cdef rtcg.RTCGeometry mesh
    cdef Vertex* vertices
    cdef Triangle* indices
    cdef unsigned int meshID

    def __init__(self, np.ndarray vertices, np.ndarray indices, rtc.EmbreeDevice device=None):
        if device is None:
            device = rtc.EmbreeDevice()
        self.device = device
        rtc.rtcSetDeviceErrorFunction(device.device, error_printer, NULL)
        self.scene_i = rtcs.rtcNewScene(device.device)
        self._build_from_indices(vertices, indices)
        rtcs.rtcCommitScene(self.scene_i)

    def set_build_quality(self, rtc.RTCBuildQuality quality):
        rtcs.rtcSetSceneBuildQuality(self.scene_i, quality)

    def get_flags(self):
        return rtcs.rtcGetSceneFlags(self.scene_i)

    def set_flags(self, int flags):
        rtcs.rtcSetSceneFlags(self.scene_i, flags)

    def __dealloc__(self):
        rtcs.rtcReleaseScene(self.scene_i)
        rtcg.rtcReleaseGeometry(self.mesh)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _build_from_indices(self, np.ndarray[np.float64_t, ndim=2] tri_vertices,
                                   np.ndarray[np.int32_t, ndim=2] tri_indices,
                                   rtc.RTCBuildQuality build_quality=rtc.RTC_BUILD_QUALITY_MEDIUM):
        cdef int i
        cdef int nv = tri_vertices.shape[0]
        cdef int nt = tri_indices.shape[0]
        cdef rtcg.RTCGeometry mesh = rtcg.rtcNewGeometry(self.device.device, rtcg.RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcg.rtcSetGeometryBuildQuality(mesh, build_quality)
        cdef Vertex* vertices = <Vertex*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_VERTEX,
                                            0, rtc.RTC_FORMAT_FLOAT3, sizeof(Vertex), nv)
        for i in range(nv):
            vertices[i].x = tri_vertices[i, 0]
            vertices[i].y = tri_vertices[i, 1]
            vertices[i].z = tri_vertices[i, 2]
        cdef Triangle* triangles = <Triangle*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_INDEX,
                                            0, rtc.RTC_FORMAT_UINT3, sizeof(Triangle), nt)
        for i in range(nt):
            triangles[i].v0 = tri_indices[i][0]
            triangles[i].v1 = tri_indices[i][1]
            triangles[i].v2 = tri_indices[i][2]
        rtcg.rtcCommitGeometry(mesh)
        cdef unsigned int meshID = rtcs.rtcAttachGeometry(self.scene_i, mesh)
        self.vertices = vertices
        self.indices = triangles
        self.meshID = meshID
        self.mesh = mesh

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def run(self, np.ndarray[np.float32_t, ndim=2] vec_origins,
            np.ndarray[np.float32_t, ndim=2] vec_directions):
        cdef int nv = vec_origins.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1] intersect_ids = np.empty(nv, dtype="int32")
        cdef rtcr.RTCIntersectContext context
        rtcr.rtcInitIntersectContext(&context)
        cdef rtcr.RTCIntersectArguments intersect_args
        rtcr.rtcInitIntersectArguments(&intersect_args)
        cdef rtcr.RTCRayHit ray_hit
        for i in range(nv):
            ray_hit.ray.org_x = vec_origins[i, 0]
            ray_hit.ray.org_y = vec_origins[i, 1]
            ray_hit.ray.org_z = vec_origins[i, 2]
            ray_hit.ray.dir_x = vec_directions[i, 0]
            ray_hit.ray.dir_y = vec_directions[i, 1]
            ray_hit.ray.dir_z = vec_directions[i, 2]
            ray_hit.ray.time = 0
            ray_hit.ray.mask = -1
            ray_hit.ray.flags = 0
            ray_hit.ray.tnear = 0.0
            ray_hit.ray.tfar = 1e37
            ray_hit.ray.id = i
            ray_hit.hit.geomID = rtcg.RTC_INVALID_GEOMETRY_ID
            ray_hit.hit.primID = rtcg.RTC_INVALID_GEOMETRY_ID
            rtcs.rtcIntersect1(self.scene_i, &context, &ray_hit, &intersect_args)
            intersect_ids[i] = ray_hit.hit.primID
        return intersect_ids

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def shortest_distance(self, np.ndarray[np.float32_t, ndim=2] vec_origins,
                          np.ndarray[np.float32_t, ndim=2] vec_directions, float offset=0.0):
        cdef int nv = vec_origins.shape[0]
        cdef np.float32_t dist = 1e37
        cdef rtcr.RTCIntersectContext context
        rtcr.rtcInitIntersectContext(&context)
        cdef rtcr.RTCIntersectArguments intersect_args
        rtcr.rtcInitIntersectArguments(&intersect_args)
        cdef rtcr.RTCRayHit ray_hit
        for i in range(nv):
            ray_hit.ray.org_x = vec_origins[i, 0] + offset * vec_directions[i, 0]
            ray_hit.ray.org_y = vec_origins[i, 1] + offset * vec_directions[i, 1]
            ray_hit.ray.org_z = vec_origins[i, 2] + offset * vec_directions[i, 2]
            ray_hit.ray.dir_x = vec_directions[i, 0]
            ray_hit.ray.dir_y = vec_directions[i, 1]
            ray_hit.ray.dir_z = vec_directions[i, 2]
            ray_hit.ray.time = 0
            ray_hit.ray.mask = -1
            ray_hit.ray.flags = 0
            ray_hit.ray.tnear = 0.0
            ray_hit.ray.tfar = dist
            ray_hit.ray.id = i
            ray_hit.hit.geomID = rtcg.RTC_INVALID_GEOMETRY_ID
            ray_hit.hit.primID = rtcg.RTC_INVALID_GEOMETRY_ID
            rtcs.rtcIntersect1(self.scene_i, &context, &ray_hit, &intersect_args)
            if ray_hit.ray.tfar < dist:
                dist = ray_hit.ray.tfar
        return dist

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def intersections(self, np.ndarray[np.float32_t, ndim=2] vec_origins,
                      np.ndarray[np.float32_t, ndim=2] vec_directions):
        cdef int nv = vec_origins.shape[0]
        cdef np.float64_t u, v, f
        cdef Triangle t
        cdef Vertex p0, p1, p2
        cdef np.ndarray[np.int32_t, ndim=1] primID = np.empty(nv, dtype="int32")
        cdef np.ndarray[np.float64_t, ndim=2] normal = np.empty((nv, 3), dtype="float64")
        cdef np.ndarray[np.float64_t, ndim=2] loc = np.empty((nv, 3), dtype="float64")
        cdef rtcr.RTCIntersectContext context
        rtcr.rtcInitIntersectContext(&context)
        cdef rtcr.RTCIntersectArguments intersect_args
        rtcr.rtcInitIntersectArguments(&intersect_args)
        cdef rtcr.RTCRayHit ray_hit
        for i in range(nv):
            ray_hit.ray.org_x = vec_origins[i, 0]
            ray_hit.ray.org_y = vec_origins[i, 1]
            ray_hit.ray.org_z = vec_origins[i, 2]
            ray_hit.ray.dir_x = vec_directions[i, 0]
            ray_hit.ray.dir_y = vec_directions[i, 1]
            ray_hit.ray.dir_z = vec_directions[i, 2]
            ray_hit.ray.time = 0
            ray_hit.ray.mask = -1
            ray_hit.ray.flags = 0
            ray_hit.ray.tnear = 0.0
            ray_hit.ray.tfar = 1e37
            ray_hit.ray.id = i
            ray_hit.hit.geomID = rtcg.RTC_INVALID_GEOMETRY_ID
            ray_hit.hit.primID = rtcg.RTC_INVALID_GEOMETRY_ID
            rtcs.rtcIntersect1(self.scene_i, &context, &ray_hit, &intersect_args)
            primID[i] = ray_hit.hit.primID
            if primID[i] >= 0:
                t = self.indices[primID[i]]
                p0 = self.vertices[t.v0]
                p1 = self.vertices[t.v1]
                p2 = self.vertices[t.v2]
                u = ray_hit.hit.u
                v = ray_hit.hit.v
                loc[i, 0] = p0.x + u * (p1.x - p0.x) + v * (p2.x - p0.x)
                loc[i, 1] = p0.y + u * (p1.y - p0.y) + v * (p2.y - p0.y)
                loc[i, 2] = p0.z + u * (p1.z - p0.z) + v * (p2.z - p0.z)
                f = 1.0 / ((ray_hit.hit.Ng_x**2 + ray_hit.hit.Ng_y**2 + ray_hit.hit.Ng_z**2)**0.5)
                normal[i, 0] = f * ray_hit.hit.Ng_x
                normal[i, 1] = f * ray_hit.hit.Ng_y
                normal[i, 2] = f * ray_hit.hit.Ng_z
        return {'primID': primID, 'normal': normal, 'loc': loc}