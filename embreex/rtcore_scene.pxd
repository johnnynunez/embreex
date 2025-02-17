# rtcore_scene.pxd wrapper for Embree 4

cimport cython
cimport numpy as np
from libcpp cimport bool
from libc.string cimport memset

from . cimport rtcore as rtc
from . cimport rtcore_ray as rtcr
from . cimport rtcore_geometry as rtcg
from . cimport rtcore_buffer as rtcb

# Use the Embree 4 header.
cdef extern from "embree4/rtcore_scene.h":
    # These types come from the header.
    ctypedef struct RTCRay
    ctypedef struct RTCRay4
    ctypedef struct RTCRay8
    ctypedef struct RTCRay16

    ctypedef struct RTCRayHit
    ctypedef struct RTCRayHit4
    ctypedef struct RTCRayHit8
    ctypedef struct RTCRayHit16

    # Scene flags – note that the old CONTEXT_FILTER flag is gone.
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

    # RTCScene is an opaque pointer.
    ctypedef void * RTCScene

    cdef struct RTCBounds
    cdef struct RTCLinearBounds

    /* Scene creation and management */
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

    ctypedef bool (*RTCProgressMonitorFunction)(void * ptr, double n);
    void rtcSetSceneProgressMonitorFunction(RTCScene scene, RTCProgressMonitorFunction progress, void * ptr);

    void rtcSetSceneBuildQuality(RTCScene scene, rtc.RTCBuildQuality quality);
    void rtcSetSceneFlags(RTCScene scene, RTCSceneFlags flags);
    RTCSceneFlags rtcGetSceneFlags(RTCScene scene);

    void rtcGetSceneBounds(RTCScene scene, RTCBounds * bounds_o);
    void rtcGetSceneLinearBounds(RTCScene scene, RTCLinearBounds * bounds_o);

    /* Point query functions omitted for brevity */

    /* --- New argument structures --- */
    ctypedef struct RTCIntersectArguments:
        rtcr.RTCIntersectContextFlags flags;    /* intersection flags; note: each field ends with a semicolon; */
        rtcr.RTCFilterFunctionN filter;          /* filter callback */
    ;

    ctypedef struct RTCOccludedArguments:
        rtcr.RTCIntersectContextFlags flags;
        rtcr.RTCFilterFunctionN filter;
    ;

    /* Provide inline initializers (simply zero‐initialize the structure) */
    cdef inline void rtcInitIntersectArguments(RTCIntersectArguments* args) nogil:
        memset(args, 0, sizeof(RTCIntersectArguments))

    cdef inline void rtcInitOccludedArguments(RTCOccludedArguments* args) nogil:
        memset(args, 0, sizeof(RTCOccludedArguments))

    /* New ray query functions. Their signature now is:
       (RTCScene scene, RTCIntersectContext* context, RTCRayHit* rayhit, RTCIntersectArguments* args)
       (and similarly for occluded queries) */
    void rtcIntersect1(RTCScene scene, rtcr.RTCIntersectContext * context, rtcr.RTCRayHit * rayhit,
                       RTCIntersectArguments * args);
    void rtcIntersect4(const int * valid, RTCScene scene, rtcr.RTCIntersectContext * context, rtcr.RTCRayHit4 * rayhit,
                       RTCIntersectArguments * args);
    void rtcIntersect8(const int * valid, RTCScene scene, rtcr.RTCIntersectContext * context, rtcr.RTCRayHit8 * rayhit,
                       RTCIntersectArguments * args);
    void rtcIntersect16(const int * valid, RTCScene scene, rtcr.RTCIntersectContext * context,
                        rtcr.RTCRayHit16 * rayhit, RTCIntersectArguments * args);

    void rtcOccluded1(RTCScene scene, rtcr.RTCIntersectContext * context, rtcr.RTCRay * ray,
                      RTCOccludedArguments * args);
    void rtcOccluded4(const int * valid, RTCScene scene, rtcr.RTCIntersectContext * context, rtcr.RTCRay4 * ray,
                      RTCOccludedArguments * args);
    void rtcOccluded8(const int * valid, RTCScene scene, rtcr.RTCIntersectContext * context, rtcr.RTCRay8 * ray,
                      RTCOccludedArguments * args);
    void rtcOccluded16(const int * valid, RTCScene scene, rtcr.RTCIntersectContext * context, rtcr.RTCRay16 * ray,
                       RTCOccludedArguments * args);

    /* Collision detection */
    ctypedef struct RTCCollision:
        unsigned int geomID0;
        unsigned int primID0;
        unsigned int geomID1;
        unsigned int primID1;
    ;
    ctypedef void (*RTCCollideFunc)(void * userPtr, RTCCollision * collisions, unsigned int num_collisions);
    void rtcCollide(RTCScene scene0, RTCScene scene1, RTCCollideFunc callback, void * userPtr);

# End extern block


# Internal enum for our own use in .pyx
cdef enum rayQueryType:
    intersect = 0,
    occluded = 1,
    distance = 2
