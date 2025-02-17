# distutils: language=c++

cimport numpy as np
cimport cython
from . cimport rtcore as rtc
from . cimport rtcore_buffer as rtcb
from . cimport rtcore_ray as rtcr
from . cimport rtcore_scene as rtcs
from . cimport rtcore_geometry as rtcg
from .rtcore cimport Vertex, Triangle

cdef extern from "mesh_construction.h":
    int triangulate_hex[12][3]
    int triangulate_tetra[4][3]

cdef class TriangleMesh:
    r'''

    This class constructs a polygon mesh with triangular elements and
    adds it to the scene.

    Parameters
    ----------

    scene : EmbreeScene
        This is the scene to which the constructed polygons will be added.
    vertices : a np.ndarray of floats.
        This specifies the x, y, and z coordinates of the vertices in
        the polygon mesh. This should either have the shape
        (num_triangles, 3, 3), or the shape (num_vertices, 3), depending
        on the value of the `indices` parameter.
    indices : either None, or a np.ndarray of ints
        If None, then vertices must have the shape (num_triangles, 3, 3).
        If indices is a np.ndarray, then it must have the shape
        (num_triangles, 3), and `vertices` must have the shape
        (num_vertices, 3).

    '''

    cdef Vertex* vertices
    cdef Triangle* indices
    cdef unsigned int meshID
    cdef rtcg.RTCGeometry mesh

    def __init__(self,
        rtcs.EmbreeScene scene,
        np.ndarray vertices,
        np.ndarray indices = None,
        rtc.RTCBuildQuality build_quality=rtc.RTC_BUILD_QUALITY_MEDIUM
    ):
        if indices is None:
            self._build_from_flat(scene, vertices, build_quality)
        else:
            self._build_from_indices(scene, vertices, indices, build_quality)

    cdef void _build_from_flat(
        self,
        rtcs.EmbreeScene scene,
        np.ndarray tri_vertices,
        rtc.RTCBuildQuality build_quality=rtc.RTC_BUILD_QUALITY_MEDIUM
    ):
        cdef int i, j
        cdef int nt = tri_vertices.shape[0]
        self.mesh = rtcg.rtcNewGeometry(scene.device.device, rtcg.RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcg.rtcSetGeometryBuildQuality(self.mesh, build_quality)

        cdef Vertex* vertices = <Vertex*> rtcg.rtcSetNewGeometryBuffer(
            self.mesh,
            rtcb.RTC_BUFFER_TYPE_VERTEX,
            0,
            rtc.RTC_FORMAT_FLOAT3,
            sizeof(Vertex),
            nt * 3
        )
        for i in range(nt):
            for j in range(3):
                vertices[i*3 + j].x = tri_vertices[i,j,0]
                vertices[i*3 + j].y = tri_vertices[i,j,1]
                vertices[i*3 + j].z = tri_vertices[i,j,2]

        cdef Triangle* triangles = <Triangle*> rtcg.rtcSetNewGeometryBuffer(
            self.mesh,
            rtcb.RTC_BUFFER_TYPE_INDEX,
            0,
            rtc.RTC_FORMAT_UINT3,
            sizeof(Triangle),
            nt
        )
        for i in range(nt):
            triangles[i].v0 = i*3 + 0
            triangles[i].v1 = i*3 + 1
            triangles[i].v2 = i*3 + 2

        rtcg.rtcCommitGeometry(self.mesh)
        self.meshID = rtcs.rtcAttachGeometry(<rtcs.RTCScene>scene.scene_i, self.mesh)
        self.vertices = vertices
        self.indices = triangles

    cdef void _build_from_indices(
        self,
        rtcs.EmbreeScene scene,
        np.ndarray tri_vertices,
        np.ndarray tri_indices,
        rtc.RTCBuildQuality build_quality=rtc.RTC_BUILD_QUALITY_MEDIUM
    ):
        cdef int i
        cdef int nv = tri_vertices.shape[0]
        cdef int nt = tri_indices.shape[0]

        self.mesh = rtcg.rtcNewGeometry(scene.device.device, rtcg.RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcg.rtcSetGeometryBuildQuality(self.mesh, build_quality)

        cdef Vertex* vertices = <Vertex*> rtcg.rtcSetNewGeometryBuffer(
            self.mesh,
            rtcb.RTC_BUFFER_TYPE_VERTEX,
            0,
            rtc.RTC_FORMAT_FLOAT3,
            sizeof(Vertex),
            nv
        )
        for i in range(nv):
            vertices[i].x = tri_vertices[i, 0]
            vertices[i].y = tri_vertices[i, 1]
            vertices[i].z = tri_vertices[i, 2]

        cdef Triangle* triangles = <Triangle*> rtcg.rtcSetNewGeometryBuffer(
            self.mesh,
            rtcb.RTC_BUFFER_TYPE_INDEX,
            0,
            rtc.RTC_FORMAT_UINT3,
            sizeof(Triangle),
            nt
        )
        for i in range(nt):
            triangles[i].v0 = tri_indices[i][0]
            triangles[i].v1 = tri_indices[i][1]
            triangles[i].v2 = tri_indices[i][2]

        rtcg.rtcCommitGeometry(self.mesh)
        self.meshID = rtcs.rtcAttachGeometry(<rtcs.RTCScene>scene.scene_i, self.mesh)
        self.vertices = vertices
        self.indices = triangles

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_vertices(self, np.ndarray[np.float32_t, ndim=2] new_vertices):
        """
        Updates the vertices of the mesh and commits the changes.
        """
        cdef int i = 0
        cdef int nv = new_vertices.shape[0]
        cdef Vertex* vertices = <Vertex*>rtcg.rtcGetGeometryBufferData(self.mesh, rtcb.RTC_BUFFER_TYPE_VERTEX, 0)
        for i in range(nv):
            vertices[i].x = new_vertices[i, 0]
            vertices[i].y = new_vertices[i, 1]
            vertices[i].z = new_vertices[i, 2]

        rtcg.rtcUpdateGeometryBuffer(self.mesh, rtcb.RTC_BUFFER_TYPE_VERTEX, 0)
        rtcg.rtcCommitGeometry(self.mesh)

    @property
    def mesh_id(self):
        return self.meshID

    def __dealloc__(self):
        rtcg.rtcReleaseGeometry(self.mesh)

cdef class ElementMesh(TriangleMesh):
    r'''

    Converts non-triangular meshes (hexahedral or tetrahedral) to triangular meshes.

    Parameters
    ----------
    scene : EmbreeScene
        Scene to add the mesh to.
    vertices : np.ndarray
        Vertex positions.
    indices : np.ndarray
        Element connectivity, shape (num_elements, 8) for hexahedral or (num_elements, 4) for tetrahedral.

    '''

    def __init__(
        self,
        rtcs.EmbreeScene scene,
        np.ndarray vertices,
        np.ndarray indices,
        rtc.RTCBuildQuality build_quality=rtc.RTC_BUILD_QUALITY_MEDIUM
    ):
        if indices.shape[1] == 8:
            self._build_from_hexahedra(scene, vertices, indices, build_quality)
        elif indices.shape[1] == 4:
            self._build_from_tetrahedra(scene, vertices, indices, build_quality)
        else:
            raise NotImplementedError("Only hexahedral or tetrahedral meshes are supported.")

    cdef void _build_from_hexahedra(
        self,
        rtcs.EmbreeScene scene,
        np.ndarray quad_vertices,
        np.ndarray quad_indices,
        rtc.RTCBuildQuality build_quality=rtc.RTC_BUILD_QUALITY_MEDIUM
    ):
        cdef int i, j
        cdef int nv = quad_vertices.shape[0]
        cdef int ne = quad_indices.shape[0]
        cdef int nt = 12 * ne  # 12 triangles per hexahedron

        self.mesh = rtcg.rtcNewGeometry(scene.device.device, rtcg.RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcg.rtcSetGeometryBuildQuality(self.mesh, build_quality)

        cdef Vertex* vertices = <Vertex*> rtcg.rtcSetNewGeometryBuffer(
            self.mesh,
            rtcb.RTC_BUFFER_TYPE_VERTEX,
            0,
            rtc.RTC_FORMAT_FLOAT3,
            sizeof(Vertex),
            nv
        )
        for i in range(nv):
            vertices[i].x = quad_vertices[i, 0]
            vertices[i].y = quad_vertices[i, 1]
            vertices[i].z = quad_vertices[i, 2]

        cdef Triangle* triangles = <Triangle*> rtcg.rtcSetNewGeometryBuffer(
            self.mesh,
            rtcb.RTC_BUFFER_TYPE_INDEX,
            0,
            rtc.RTC_FORMAT_UINT3,
            sizeof(Triangle),
            nt
        )
        for i in range(ne):
            for j in range(12):
                triangles[12*i+j].v0 = quad_indices[i][triangulate_hex[j][0]]
                triangles[12*i+j].v1 = quad_indices[i][triangulate_hex[j][1]]
                triangles[12*i+j].v2 = quad_indices[i][triangulate_hex[j][2]]

        rtcg.rtcCommitGeometry(self.mesh)
        self.meshID = rtcs.rtcAttachGeometry(<rtcs.RTCScene>scene.scene_i, self.mesh)
        self.vertices = vertices
        self.indices = triangles

    cdef void _build_from_tetrahedra(
        self,
        rtcs.EmbreeScene scene,
        np.ndarray tetra_vertices,
        np.ndarray tetra_indices,
        rtc.RTCBuildQuality build_quality=rtc.RTC_BUILD_QUALITY_MEDIUM
    ):
        cdef int i, j
        cdef int nv = tetra_vertices.shape[0]
        cdef int ne = tetra_indices.shape[0]
        cdef int nt = 4 * ne  # 4 triangles per tetrahedron

        self.mesh = rtcg.rtcNewGeometry(scene.device.device, rtcg.RTC_GEOMETRY_TYPE_TRIANGLE)
        rtcg.rtcSetGeometryBuildQuality(self.mesh, build_quality)

        cdef Vertex* vertices = <Vertex*> rtcg.rtcSetNewGeometryBuffer(
            self.mesh,
            rtcb.RTC_BUFFER_TYPE_VERTEX,
            0,
            rtc.RTC_FORMAT_FLOAT3,
            sizeof(Vertex),
            nv
        )
        for i in range(nv):
            vertices[i].x = tetra_vertices[i, 0]
            vertices[i].y = tetra_vertices[i, 1]
            vertices[i].z = tetra_vertices[i, 2]

        cdef Triangle* triangles = <Triangle*> rtcg.rtcSetNewGeometryBuffer(
            self.mesh,
            rtcb.RTC_BUFFER_TYPE_INDEX,
            0,
            rtc.RTC_FORMAT_UINT3,
            sizeof(Triangle),
            nt
        )
        for i in range(ne):
            for j in range(4):
                triangles[4*i+j].v0 = tetra_indices[i][triangulate_tetra[j][0]]
                triangles[4*i+j].v1 = tetra_indices[i][triangulate_tetra[j][1]]
                triangles[4*i+j].v2 = tetra_indices[i][triangulate_tetra[j][2]]

        rtcg.rtcCommitGeometry(self.mesh)
        self.meshID = rtcs.rtcAttachGeometry(<rtcs.RTCScene>scene.scene_i, self.mesh)
        self.vertices = vertices
        self.indices = triangles