# rtcore_scene.pxd wrapper

cimport cython
from numpy cimport int32_t, float32_t, float64_t
cimport numpy as np
from libcpp cimport bool
from . cimport rtcore as rtc
from . cimport rtcore_ray as rtcr    # Importa estructuras de ray/hit/context
from . cimport rtcore_geometry as rtcg
from . cimport rtcore_buffer as rtcb  # Agregado

# RTCBounds y RTCLinearBounds se importan desde rtcore.pxd
from .rtcore cimport RTCBounds, RTCLinearBounds

# Funciones y estructuras del header embree4/rtcore_scene.h
cdef extern from "embree4/rtcore_scene.h":
    # Declaraciones básicas de ray y hit
    ctypedef struct RTCRay
    ctypedef struct RTCRay4
    ctypedef struct RTCRay8
    ctypedef struct RTCRay16

    ctypedef struct RTCRayHit
    ctypedef struct RTCRayHit4
    ctypedef struct RTCRayHit8
    ctypedef struct RTCRayHit16

    # Scene flags (el flag de context filter ha desaparecido)
    cpdef enum RTCSceneFlags:
        RTC_SCENE_FLAG_NONE
        RTC_SCENE_FLAG_DYNAMIC
        RTC_SCENE_FLAG_COMPACT
        RTC_SCENE_FLAG_ROBUST

    cdef enum RTCAlgorithmFlags:
        RTC_INTERSECT1
        RTC_INTERSECT4
        RTC_INTERSECT8
        RTC_INTERSECT16

    # RTCScene; RTCDevice ya está definido en rtcore.pxd
    ctypedef void* RTCScene

    cdef struct RTCBounds
    cdef struct RTCLinearBounds

    # Funciones de manejo de escenas
    RTCScene rtcNewScene(rtc.RTCDevice device);
    rtc.RTCDevice rtcGetSceneDevice(RTCScene hscene);
    void rtcRetainScene(RTCScene scene);
    void rtcReleaseScene(RTCScene scene);

    unsigned int rtcAttachGeometry(RTCScene scene, rtcg.RTCGeometry geometry);
    void rtcAttachGeometryByID(RTCScene scene, rtcg.RTCGeometry geometry, unsigned int geomID);
    void rtcDetachGeometry(RTCScene scene, unsigned int geomID);
    rtcg.RTCGeometry rtcGetGeometry(RTCScene scene, unsigned int geomID);
    rtcg.RTCGeometry rtcGetGeometryThreadSafe(RTCScene scene, unsigned int geomID);

    void rtcCommitScene(RTCScene scene);
    void rtcJoinCommitScene(RTCScene scene);

    ctypedef bool (*RTCProgressMonitorFunction)(void* ptr, double n);
    void rtcSetSceneProgressMonitorFunction(RTCScene scene, RTCProgressMonitorFunction progress, void* ptr);

    void rtcSetSceneBuildQuality(RTCScene scene, rtc.RTCBuildQuality quality);
    void rtcSetSceneFlags(RTCScene scene, RTCSceneFlags flags);
    RTCSceneFlags rtcGetSceneFlags(RTCScene scene);

    void rtcGetSceneBounds(RTCScene scene, RTCBounds* bounds_o);
    void rtcGetSceneLinearBounds(RTCScene scene, RTCLinearBounds* bounds_o);

    # Argument structs para intersección y oclusión
    ctypedef struct RTCIntersectArguments:
        rtcr.RTCIntersectContextFlags flags
        rtcr.RTCFilterFunctionN filter

    ctypedef struct RTCOccludedArguments:
        rtcr.RTCIntersectContextFlags flags
        rtcr.RTCFilterFunctionN filter

    void rtcInitIntersectArguments(RTCIntersectArguments* args);
    void rtcInitOccludedArguments(RTCOccludedArguments* args);

    # Funciones de ray query
    void rtcIntersect1(RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRayHit* rayhit, RTCIntersectArguments* args);
    void rtcIntersect4(const int* valid, RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRayHit4* rayhit, RTCIntersectArguments* args);
    void rtcIntersect8(const int* valid, RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRayHit8* rayhit, RTCIntersectArguments* args);
    void rtcIntersect16(const int* valid, RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRayHit16* rayhit, RTCIntersectArguments* args);

    void rtcIntersect1M(RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRayHit* rayhit, unsigned int M, size_t byteStride);
    void rtcIntersect1Mp(RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRayHit** rayhit, unsigned int M);
    void rtcIntersectNM(RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRayHitN* rayhit, unsigned int N, unsigned int M, size_t byteStride);
    void rtcIntersectNp(RTCScene scene, rtcr.RTCIntersectContext* context, const rtcr.RTCRayHitNp* rayhit, unsigned int N);

    void rtcOccluded1(RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRay* ray, RTCOccludedArguments* args);
    void rtcOccluded4(const int* valid, RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRay4* ray, RTCOccludedArguments* args);
    void rtcOccluded8(const int* valid, RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRay8* ray, RTCOccludedArguments* args);
    void rtcOccluded16(const int* valid, RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRay16* ray, RTCOccludedArguments* args);
    void rtcOccluded1M(RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRay* ray, unsigned int M, size_t byteStride);
    void rtcOccluded1Mp(RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRay** ray, unsigned int M);
    void rtcOccludedNM(RTCScene scene, rtcr.RTCIntersectContext* context, rtcr.RTCRayN* ray, unsigned int N, unsigned int M, size_t byteStride);
    void rtcOccludedNp(RTCScene scene, rtcr.RTCIntersectContext* context, const rtcr.RTCRayNp* ray, unsigned int N);

    # Collision detection (opcional)
    ctypedef struct RTCCollision:
        unsigned int geomID0
        unsigned int primID0
        unsigned int geomID1
        unsigned int primID1

    ctypedef void (*RTCCollideFunc)(void* userPtr, RTCCollision* collisions, unsigned int num_collisions);
    void rtcCollide(RTCScene scene0, RTCScene scene1, RTCCollideFunc callback, void* userPtr);

# Declaración de la clase de escena para que otros módulos la puedan usar.
cdef class EmbreeScene:
    cdef RTCScene scene_i
    cdef public int is_committed
    cdef rtc.EmbreeDevice device

# Nuestro enum para ray query (para uso interno)
cdef enum rayQueryType:
    intersect = 0,
    occluded = 1,
    distance = 2