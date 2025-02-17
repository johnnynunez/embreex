# distutils: language=c++

cimport cython
cimport numpy as np
from numpy cimport float32_t, float64_t, int32_t

# Importaciones de definiciones de Embree
from .rtcore cimport RTC_FORMAT_FLOAT3, RTC_FORMAT_UINT3, Vertex, Triangle, RTCDevice
from .rtcore_geometry cimport RTCBuildQuality, RTC_GEOMETRY_TYPE_TRIANGLE
from .rtcore_buffer cimport RTC_BUFFER_TYPE_VERTEX, RTC_BUFFER_TYPE_INDEX
from .rtcore_scene cimport EmbreeScene, RTCScene

# Importamos las tablas de triangulación desde el header correspondiente
cdef extern from "mesh_construction.h":
    int triangulate_hex[12][3]
    int triangulate_tetra[4][3]

###############################################################################
# Clase TriangleMesh
###############################################################################
cdef class TriangleMesh:
    """
    Construye una malla triangular y la agrega a la escena.

    Parámetros
    ----------
    scene : EmbreeScene
        Escena de Embree.
    vertices : np.ndarray
        Si indices es None, se espera (num_triángulos, 3, 3) (cada triángulo con
        sus tres vértices). Si se usa indices, debe ser (num_vertices, 3).
    indices : np.ndarray o None
        Si es None se asume que 'vertices' ya contiene los 3 vértices de cada triángulo;
        en caso contrario, debe ser (num_triángulos, 3).
    build_quality : int
        Valor de la enumeración RTCBuildQuality (por ejemplo, RTC_BUILD_QUALITY_MEDIUM).
    """
    cdef Vertex * vertices
    cdef Triangle * indices
    cdef unsigned int meshID
    cdef RTCGeometry mesh

    def __init__(self, EmbreeScene scene, np.ndarray vertices, np.ndarray indices=None,
                 int build_quality=RTCBuildQuality.RTC_BUILD_QUALITY_MEDIUM):
        if indices is None:
            self._build_from_flat(scene, vertices, build_quality)
        else:
            self._build_from_indices(scene, vertices, indices, build_quality)

    cdef void _build_from_flat(self, EmbreeScene scene, np.ndarray tri_vertices, int build_quality):
        cdef int i, j
        cdef int nt = tri_vertices.shape[0]

        # Crear nueva geometría de triángulos
        self.mesh = rtcNewGeometry(scene.device.device, RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcSetGeometryBuildQuality(self.mesh, <RTCBuildQuality> build_quality)

        # Reservar buffer para vértices
        cdef Vertex * verts = <Vertex *> rtcSetNewGeometryBuffer(self.mesh, RTC_BUFFER_TYPE_VERTEX, 0,
                                                                 RTC_FORMAT_FLOAT3, sizeof(Vertex), nt * 3)
        for i in range(nt):
            for j in range(3):
                verts[i * 3 + j].x = tri_vertices[i, j, 0]
                verts[i * 3 + j].y = tri_vertices[i, j, 1]
                verts[i * 3 + j].z = tri_vertices[i, j, 2]

        # Reservar buffer para índices
        cdef Triangle * tris = <Triangle *> rtcSetNewGeometryBuffer(self.mesh, RTC_BUFFER_TYPE_INDEX, 0,
                                                                    RTC_FORMAT_UINT3, sizeof(Triangle), nt)
        for i in range(nt):
            tris[i].v0 = i * 3 + 0
            tris[i].v1 = i * 3 + 1
            tris[i].v2 = i * 3 + 2

        rtcCommitGeometry(self.mesh)
        self.meshID = rtcAttachGeometry(<RTCScene> scene.scene_i, self.mesh)
        self.vertices = verts
        self.indices = tris

    cdef void _build_from_indices(self, EmbreeScene scene, np.ndarray tri_vertices, np.ndarray tri_indices,
                                  int build_quality):
        cdef int i
        cdef int nv = tri_vertices.shape[0]
        cdef int nt = tri_indices.shape[0]

        # Crear nueva geometría de triángulos
        self.mesh = rtcNewGeometry(scene.device.device, RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcSetGeometryBuildQuality(self.mesh, <RTCBuildQuality> build_quality)

        # Reservar buffer para vértices
        cdef Vertex * verts = <Vertex *> rtcSetNewGeometryBuffer(self.mesh, RTC_BUFFER_TYPE_VERTEX, 0,
                                                                 RTC_FORMAT_FLOAT3, sizeof(Vertex), nv)
        for i in range(nv):
            verts[i].x = tri_vertices[i, 0]
            verts[i].y = tri_vertices[i, 1]
            verts[i].z = tri_vertices[i, 2]

        # Reservar buffer para índices
        cdef Triangle * tris = <Triangle *> rtcSetNewGeometryBuffer(self.mesh, RTC_BUFFER_TYPE_INDEX, 0,
                                                                    RTC_FORMAT_UINT3, sizeof(Triangle), nt)
        for i in range(nt):
            tris[i].v0 = tri_indices[i][0]
            tris[i].v1 = tri_indices[i][1]
            tris[i].v2 = tri_indices[i][2]

        rtcCommitGeometry(self.mesh)
        self.meshID = rtcAttachGeometry(<RTCScene> scene.scene_i, self.mesh)
        self.vertices = verts
        self.indices = tris

    def update_vertices(self, np.ndarray[np.float32_t, ndim=2] new_vertices):
        """
        Actualiza los vértices y recompila la geometría.
        """
        cdef int i, nv = new_vertices.shape[0]
        cdef Vertex * verts = <Vertex *> rtcGetGeometryBufferData(self.mesh, RTC_BUFFER_TYPE_VERTEX, 0)
        for i in range(nv):
            verts[i].x = new_vertices[i, 0]
            verts[i].y = new_vertices[i, 1]
            verts[i].z = new_vertices[i, 2]
        rtcUpdateGeometryBuffer(self.mesh, RTC_BUFFER_TYPE_VERTEX, 0)
        rtcCommitGeometry(self.mesh)

    property mesh_id:
        def __get__(self):
            return self.meshID

    def __dealloc__(self):
        rtcReleaseGeometry(self.mesh)

# Note: The `ElementMesh` class would need similar updates, but since it inherits from `TriangleMesh`