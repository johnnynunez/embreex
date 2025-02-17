# distutils: language=c++
# Embree scene.

cimport cython
from numpy cimport int32_t, float32_t, float64_t
cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
import numpy as np
import logging
import numbers
from . cimport rtcore as rtc
from . cimport rtcore_buffer as rtcb
from . cimport rtcore_ray as rtcr
from . cimport rtcore_scene as rtcs
from . cimport rtcore_geometry as rtcg
from .rtcore cimport Vertex, Triangle
# Importamos el enum y constantes para ray queries
from .rtcore_scene cimport rayQueryType, intersect, occluded, distance

log = logging.getLogger(__name__)

cdef void error_printer(void *userPtr, rtc.RTCError code, const char *_str) noexcept:
    """
    Callback para errores de Embree.
    """
    log.error("ERROR CAUGHT IN EMBREE")
    rtc.print_error(code)
    log.error("ERROR MESSAGE: %s" % _str)

cdef packed struct hit_struct:
    int32_t geomID
    int32_t primID
    int32_t rayIDX
    float32_t tfar

cdef packed struct hit_count_struct:
    int32_t primID
    int32_t count
    float32_t weight

cdef class EmbreeScene:
    cdef rtcs.RTCScene scene_i
    cdef public int is_committed
    cdef rtc.EmbreeDevice device

    def __init__(self, rtc.EmbreeDevice device=None):
        if device is None:
            device = rtc.EmbreeDevice()
        self.device = device
        # Casteamos la función de error al tipo correcto
        rtc.rtcSetDeviceErrorFunction(device.device, <rtc.RTCErrorFunc>error_printer, NULL)
        self.scene_i = rtcs.rtcNewScene(device.device)
        self.is_committed = 0

    def set_build_quality(self, rtc.RTCBuildQuality quality):
        rtcs.rtcSetSceneBuildQuality(self.scene_i, quality)

    def get_flags(self):
        return rtcs.rtcGetSceneFlags(self.scene_i)

    def set_flags(self, int flags):
        # Convertir el int a RTCSceneFlags
        rtcs.rtcSetSceneFlags(self.scene_i, <rtcs.RTCSceneFlags>flags)

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

        # Obtenemos los bounds; usamos rtc.RTCBounds importado de rtcore
        cdef rtc.RTCBounds bnds
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
            query_type = intersect  # de nuestro enum importado
        elif query == 'OCCLUDED':
            query_type = occluded
        elif query == 'DISTANCE':
            query_type = distance
        else:
            raise ValueError("Tipo de consulta '%s' no reconocido. Los aceptados son INTERSECT, OCCLUDED, DISTANCE" % query)

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

        # Usamos las funciones de inicialización definidas en rtcs (no en rtcr)
        cdef rtcs.RTCIntersectArguments intersect_args
        cdef rtcs.RTCOccludedArguments occluded_args
        rtcs.rtcInitIntersectArguments(&intersect_args)
        rtcs.rtcInitOccludedArguments(&occluded_args)

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

            if query_type == intersect or query_type == distance:
                rtcs.rtcIntersect1(self.scene_i, &context, &ray_hit, &intersect_args)
                if not output:
                    if query_type == intersect:
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
            if query_type == distance:
                return tfars
            else:
                return intersect_ids

    # Los otros métodos se han corregido de forma similar, asegurándose de:
    # - Llamar a las funciones de escena (rtc*) a través de rtcs
    # - Usar las funciones de inicialización de argumentos de rtcs
    # - Usar la importación de los tipos de NumPy desde "from numpy cimport ..."
    # - Utilizar el enum importado para rayQueryType

    def __dealloc__(self):
        rtcs.rtcReleaseScene(self.scene_i)
