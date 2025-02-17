# distutils: language=c++

cimport cython
cimport numpy as np
from numpy cimport float32_t, float64_t, int32_t

# Importar constantes y tipos básicos de Embree desde rtcore
from .rtcore cimport RTC_BUILD_QUALITY_MEDIUM, RTC_FORMAT_FLOAT3, RTC_FORMAT_UINT3, Vertex, Triangle, RTCDevice

# Importamos el módulo de escena asignándole el alias rtcs
cimport rtcore_scene as rtcs

# Importamos los módulos de buffer, ray y geometría
from . cimport rtcore_buffer as rtcb
from . cimport rtcore_ray as rtcr
from . cimport rtcore_geometry as rtcg

# Importar las constantes de geometría (por ejemplo, RTC_GEOMETRY_TYPE_TRIANGLE)
from .rtcore_geometry cimport RTC_GEOMETRY_TYPE_TRIANGLE

# Importar arrays de triangulación desde el header
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
        La escena a la que se agregará la malla.
    vertices : np.ndarray
        Si no se usan índices, se espera la forma (num_triángulos, 3, 3)
        (cada triángulo con sus tres vértices). Si se usan índices, la forma
        debe ser (num_vertices, 3).
    indices : np.ndarray o None
        Si es None se asume que 'vertices' ya contiene las coordenadas de cada
        triángulo (duplicando vértices compartidos). Si se especifica, debe tener
        forma (num_triángulos, 3) y los vértices se extraerán de un array independiente.
    build_quality : int
        Calidad de construcción (valor de la enumeración RTCBuildQuality),
        por defecto RTC_BUILD_QUALITY_MEDIUM.
    """
    cdef Vertex* vertices
    cdef Triangle* indices
    cdef unsigned int meshID
    cdef rtcg.RTCGeometry mesh

    def __init__(self, rtcs.EmbreeScene scene, np.ndarray vertices, np.ndarray indices = None,
                 int build_quality = RTC_BUILD_QUALITY_MEDIUM):
        if indices is None:
            self._build_from_flat(scene, vertices, build_quality)
        else:
            self._build_from_indices(scene, vertices, indices, build_quality)

    cdef void _build_from_flat(self, rtcs.EmbreeScene scene, np.ndarray tri_vertices, int build_quality):
        cdef int i, j
        cdef int nt = tri_vertices.shape[0]
        # Crear nueva geometría de triángulos
        cdef rtcg.RTCGeometry mesh = rtcg.rtcNewGeometry(scene.device.device, RTC_GEOMETRY_TYPE_TRIANGLE)
        # Se realiza un cast al tipo correcto de build quality
        rtcg.rtcSetGeometryBuildQuality(mesh, <rtcs.RTCBuildQuality>build_quality)
        cdef Vertex* verts = <Vertex*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_VERTEX, 0,
                                        RTC_FORMAT_FLOAT3, sizeof(Vertex), nt * 3)
        for i in range(nt):
            for j in range(3):
                verts[i*3 + j].x = tri_vertices[i, j, 0]
                verts[i*3 + j].y = tri_vertices[i, j, 1]
                verts[i*3 + j].z = tri_vertices[i, j, 2]
        cdef Triangle* tris = <Triangle*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_INDEX, 0,
                                        RTC_FORMAT_UINT3, sizeof(Triangle), nt)
        for i in range(nt):
            tris[i].v0 = i*3 + 0
            tris[i].v1 = i*3 + 1
            tris[i].v2 = i*3 + 2
        rtcg.rtcCommitGeometry(mesh)
        # Convertir la escena al tipo RTCScene (definido como void*)
        cdef rtcs.RTCScene sc = <rtcs.RTCScene>scene.scene_i
        cdef unsigned int meshID = rtcs.rtcAttachGeometry(<void*>sc, mesh)
        self.vertices = verts
        self.indices = tris
        self.meshID = meshID
        self.mesh = mesh

    cdef void _build_from_indices(self, rtcs.EmbreeScene scene, np.ndarray tri_vertices, np.ndarray tri_indices,
                                   int build_quality):
        cdef int i
        cdef int nv = tri_vertices.shape[0]
        cdef int nt = tri_indices.shape[0]
        cdef rtcg.RTCGeometry mesh = rtcg.rtcNewGeometry(scene.device.device, RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcg.rtcSetGeometryBuildQuality(mesh, <rtcs.RTCBuildQuality>build_quality)
        cdef Vertex* verts = <Vertex*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_VERTEX, 0,
                                        RTC_FORMAT_FLOAT3, sizeof(Vertex), nv)
        for i in range(nv):
            verts[i].x = tri_vertices[i, 0]
            verts[i].y = tri_vertices[i, 1]
            verts[i].z = tri_vertices[i, 2]
        cdef Triangle* tris = <Triangle*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_INDEX, 0,
                                        RTC_FORMAT_UINT3, sizeof(Triangle), nt)
        for i in range(nt):
            tris[i].v0 = tri_indices[i][0]
            tris[i].v1 = tri_indices[i][1]
            tris[i].v2 = tri_indices[i][2]
        rtcg.rtcCommitGeometry(mesh)
        cdef rtcs.RTCScene sc = <rtcs.RTCScene>scene.scene_i
        cdef unsigned int meshID = rtcs.rtcAttachGeometry(<void*>sc, mesh)
        self.vertices = verts
        self.indices = tris
        self.meshID = meshID
        self.mesh = mesh

    def update_vertices(self, np.ndarray[np.float32_t, ndim=2] new_vertices):
        """
        Actualiza los vértices de la geometría y los recompila.
        """
        cdef int i, nv = new_vertices.shape[0]
        cdef Vertex* verts = <Vertex*> rtcg.rtcGetGeometryBufferData(self.mesh, rtcb.RTC_BUFFER_TYPE_VERTEX, 0)
        for i in range(nv):
            verts[i].x = new_vertices[i, 0]
            verts[i].y = new_vertices[i, 1]
            verts[i].z = new_vertices[i, 2]
        rtcg.rtcUpdateGeometryBuffer(self.mesh, rtcb.RTC_BUFFER_TYPE_VERTEX, 0)
        rtcg.rtcCommitGeometry(self.mesh)

    property mesh_id:
        def __get__(self):
            return self.meshID

    def __dealloc__(self):
        rtcg.rtcReleaseGeometry(self.mesh)

###############################################################################
# Clase ElementMesh (convierte mallas no triangulares a trianguladas)
###############################################################################
cdef class ElementMesh(TriangleMesh):
    """
    Convierte mallas hexaédricas o tetraédricas a mallas triangulares.
    Se espera que:
      - Para hexaédricas, los índices tengan forma (num_elementos, 8)
      - Para tetraédricas, tengan forma (num_elementos, 4)
    """
    def __init__(self, rtcs.EmbreeScene scene, np.ndarray vertices, np.ndarray indices,
                 int build_quality = RTC_BUILD_QUALITY_MEDIUM):
        if indices.shape[1] == 8:
            self._build_from_hexahedra(scene, vertices, indices, build_quality)
        elif indices.shape[1] == 4:
            self._build_from_tetrahedra(scene, vertices, indices, build_quality)
        else:
            raise NotImplementedError("Formato de índices no soportado.")

    cdef void _build_from_hexahedra(self, rtcs.EmbreeScene scene, np.ndarray quad_vertices,
                                     np.ndarray quad_indices, int build_quality):
        cdef int i, j
        cdef int nv = quad_vertices.shape[0]
        cdef int ne = quad_indices.shape[0]
        cdef int nt = 12 * ne  # 12 triángulos por elemento
        cdef rtcg.RTCGeometry mesh = rtcg.rtcNewGeometry(scene.device.device, RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcg.rtcSetGeometryBuildQuality(mesh, <rtcs.RTCBuildQuality>build_quality)
        cdef Vertex* verts = <Vertex*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_VERTEX, 0,
                                        RTC_FORMAT_FLOAT3, sizeof(Vertex), nv)
        for i in range(nv):
            verts[i].x = quad_vertices[i, 0]
            verts[i].y = quad_vertices[i, 1]
            verts[i].z = quad_vertices[i, 2]
        cdef Triangle* tris = <Triangle*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_INDEX, 0,
                                        RTC_FORMAT_UINT3, sizeof(Triangle), nt)
        for i in range(ne):
            for j in range(12):
                tris[12 * i + j].v0 = quad_indices[i][triangulate_hex[j][0]]
                tris[12 * i + j].v1 = quad_indices[i][triangulate_hex[j][1]]
                tris[12 * i + j].v2 = quad_indices[i][triangulate_hex[j][2]]
        rtcg.rtcCommitGeometry(mesh)
        cdef rtcs.RTCScene sc = <rtcs.RTCScene>scene.scene_i
        cdef unsigned int meshID = rtcs.rtcAttachGeometry(<void*>sc, mesh)
        self.vertices = verts
        self.indices = tris
        self.meshID = meshID
        self.mesh = mesh

    cdef void _build_from_tetrahedra(self, rtcs.EmbreeScene scene, np.ndarray tetra_vertices,
                                      np.ndarray tetra_indices, int build_quality):
        cdef int i, j
        cdef int nv = tetra_vertices.shape[0]
        cdef int ne = tetra_indices.shape[0]
        cdef int nt = 4 * ne  # 4 triángulos por tetraedro
        cdef rtcg.RTCGeometry mesh = rtcg.rtcNewGeometry(scene.device.device, RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcg.rtcSetGeometryBuildQuality(mesh, <rtcs.RTCBuildQuality>build_quality)
        cdef Vertex* verts = <Vertex*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_VERTEX, 0,
                                        RTC_FORMAT_FLOAT3, sizeof(Vertex), nv)
        for i in range(nv):
            verts[i].x = tetra_vertices[i, 0]
            verts[i].y = tetra_vertices[i, 1]
            verts[i].z = tetra_vertices[i, 2]
        cdef Triangle* tris = <Triangle*> rtcg.rtcSetNewGeometryBuffer(mesh, rtcb.RTC_BUFFER_TYPE_INDEX, 0,
                                        RTC_FORMAT_UINT3, sizeof(Triangle), nt)
        for i in range(ne):
            for j in range(4):
                tris[4*i + j].v0 = tetra_indices[i][triangulate_tetra[j][0]]
                tris[4*i + j].v1 = tetra_indices[i][triangulate_tetra[j][1]]
                tris[4*i + j].v2 = tetra_indices[i][triangulate_tetra[j][2]]
        rtcg.rtcCommitGeometry(mesh)
        cdef rtcs.RTCScene sc = <rtcs.RTCScene>scene.scene_i
        cdef unsigned int meshID = rtcs.rtcAttachGeometry(<void*>sc, mesh)
        self.vertices = verts
        self.indices = tris
        self.meshID = meshID
        self.mesh = mesh

    def __dealloc__(self):
        rtcg.rtcReleaseGeometry(self.mesh)
