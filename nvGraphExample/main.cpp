#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "nvgraph.h"

#define DEBUG false

void check(nvgraphStatus_t status) {
	if (status != NVGRAPH_STATUS_SUCCESS) {
		std::string output;
		switch (status) {
		case NVGRAPH_STATUS_NOT_INITIALIZED:
			output = "NVGRAPH_STATUS_NOT_INITIALIZED";
			break;
		case NVGRAPH_STATUS_ALLOC_FAILED:
			output = "NVGRAPH_STATUS_ALLOC_FAILED";
			break;
		case NVGRAPH_STATUS_INVALID_VALUE:
			output = "NVGRAPH_STATUS_INVALID_VALUE";
			break;
		case NVGRAPH_STATUS_ARCH_MISMATCH:
			output = "NVGRAPH_STATUS_ARCH_MISMATCH";
			break;
		case NVGRAPH_STATUS_MAPPING_ERROR:
			output = "NVGRAPH_STATUS_MAPPING_ERROR";
			break;
		case NVGRAPH_STATUS_EXECUTION_FAILED:
			output = "NVGRAPH_STATUS_EXECUTION_FAILED";
			break;
		case NVGRAPH_STATUS_INTERNAL_ERROR:
			output = "NVGRAPH_STATUS_INTERNAL_ERROR";
			break;
		case NVGRAPH_STATUS_TYPE_NOT_SUPPORTED:
			output = "NVGRAPH_STATUS_TYPE_NOT_SUPPORTED";
			break;
		case NVGRAPH_STATUS_NOT_CONVERGED:
			output = "NVGRAPH_STATUS_NOT_CONVERGED";
			break;
		}
		std::cout << output << '\n';
		system("PAUSE");
		exit(0);
	}
}

void load_from_file(std::string, int*, int*, float*, int);
void nvgraph_pagerank(nvgraphHandle_t, nvgraphGraphDescr_t, float*, float*);
void nvgraph_sssp(nvgraphHandle_t*, nvgraphGraphDescr_t*, float*, float*);
void nvgraph_convert_topology(nvgraphHandle_t, int, int, void*, void*, void*, nvgraphTopologyType_t, void*);

int main(int argc, char **argv) {
	//variables for nvGraph
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCSCTopology32I_t col_major_topology;
	cudaDataType_t edge_dimT = CUDA_R_32F;

	//specifics for the dataset being used
	int nvertices = 6, nedges = 10; //REMEMBER TO CHANGE THIS FOR EVERY NEW FILE
	int *source_indices_h, *destination_indices_h;
	float *edge_data_h;
	source_indices_h = (int *)malloc(sizeof(int)*nedges);
	destination_indices_h = (int *)malloc(sizeof(int)*nedges);
	edge_data_h = (float *)malloc(sizeof(float)*nedges);

	//initialization of nvGraph
	nvgraphCreate(&handle);
	nvgraphCreateGraphDescr(handle, &graph);

	//load the dataset from the file
	load_from_file("test-sssp.txt", source_indices_h, destination_indices_h, edge_data_h, nedges);

	//Initialize topology for conversion and convert
	col_major_topology = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));
	nvgraph_convert_topology(handle, nedges, nvertices, source_indices_h, destination_indices_h, edge_data_h, NVGRAPH_CSC_32, col_major_topology);

	//Set the graph (can only be set once)
	if (DEBUG)
		std::cout << "Setting graph structure\n";
	check(nvgraphSetGraphStructure(handle, graph, (void*)col_major_topology, NVGRAPH_CSC_32));

	//Specific to nvgraph_sssp()
	float *result = (float *)malloc(nvertices*sizeof(float));
	nvgraph_sssp(&handle, &graph, edge_data_h, result);
	for (int i = 0; i < nvertices; i++) {
		std::cout << result[i] << '\n';
	}
	free(result);
	//END of specifics for nvgraph_sssp()

	//Destroy graph handles
	if (DEBUG)
		std::cout << "Destroying graph\n";
	nvgraphDestroyGraphDescr(handle, graph);
	//Destroy and unallocate any memory allocated using this handle
	nvgraphDestroy(handle);

	//cleanup
	cudaFree(col_major_topology->source_indices);
	cudaFree(col_major_topology->destination_offsets);
	free(col_major_topology);
	free(edge_data_h);
	free(source_indices_h);
	free(destination_indices_h);

	system("PAUSE");
	return 0;
}

//Wrapper for nvgraphPagerank()
void nvgraph_pagerank(nvgraphHandle_t handle, nvgraphGraphDescr_t graph, float* edge_data, float *result) {
	int vertex_numsets = 2, edge_numsets = 1;
	cudaDataType_t edge_type = CUDA_R_32F;
	cudaDataType_t *vertex_type;

	//allocate memory for each type
	vertex_type = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
	//If types are the same, then this is unnecessary. However, if different types are needed/wanted, then
	//it is good practice to set each type with respect to the vertex data (i.e. vertex_data[0]'s type should correspond to vertex_type[0])
	vertex_type[0] = CUDA_R_32F, vertex_type[1] = CUDA_R_32F;

	//TODO: needs bookmark

	//Perform setup for algorithm
	check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_type));
	check(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_type));
	check(nvgraphSetEdgeData(handle, graph, (void*)edge_data, 0));
}

//Wrapper for nvgraphSssp()
void nvgraph_sssp(nvgraphHandle_t *handle_p, nvgraphGraphDescr_t *graph_p, float* edge_data, float* result) {
	nvgraphHandle_t handle = *handle_p; nvgraphGraphDescr_t graph = *graph_p;
	std::cout << "STARTING NVGRAPH_SSSP()\n";
	int vertex_numsets = 1, edge_numsets = 1, source_vertex = 0;
	cudaDataType_t edge_type = CUDA_R_32F, vertex_type = CUDA_R_32F;

	//Perform setup for algorithm
	if (DEBUG)
		std::cout << "performing setup\n";
	check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, &vertex_type));
	check(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_type));
	check(nvgraphSetEdgeData(handle, graph, (void*)edge_data, 0));

	//Perform algorithm
	if (DEBUG)
		std::cout << "performing algorithm\n";
	check(nvgraphSssp(handle, graph, 0, &source_vertex, 0));//(handle,graph,index of edge set, pointer to source vertex number, output index)
															/* EXPLANATION:
															HANDLE: this is the nvgraphHandle_t created in main
															GRAPH: this is the nvgraphGraphDescr_t also created in main. this also needs to have a structure set using nvgraphSetGraphStructure()
															INDEX OF EDGE SET: this is used when multiple edge sets are provided/wanted. To accomplish multiple,
															1. change "float *edge_data" to "float **edge_data", since the pointer to the list of edge data will be needed.
															2. update "edge_numsets" to be the size of the list of edge data (i.e. how many different edge sets being used)
															3. update "edge_type" to "cudaDataType_t *edge_type" and set respective types (i.e. edge_type[0] should correspond to edge_data[0])
															4. update allocation of edge data (i.e. "&edge_type" should become "edge_type")
															5. update the setting of edge data (i.e. for each edge data call nvgraphSetEdgeData() e.g. inside FOR loop: nvgraphSetEdgeData(handle, graph, (void*)edge_data[iter], iter))
															6. make edge data selection (i.e. set this value to index of whatever edge set wanted)
															7. update "float *result" to "float **results" (add the 's' for readability). this allows for multiple results to then be saved
															POINTER TO SOURCE VERTEX: this is just a pointer to an "int" where the value is the index of the source/starting vertex
															OUTPUT INDEX: this is the index of the allocated vertex data (nvgraphAllocateVertexData()). this allows for multiple executions of the algorithm without
															having to manage result storage until the end. BE SURE TO ALLOCATE THE CORRECT NUMBER OF INDICES! (i.e. if you want to store 3 different results, then allocate
															3 different vertex data sets.)
															*/

															//Save algorithm output to "result"
	if (DEBUG)
		std::cout << "acquiring results\n";
	check(nvgraphGetVertexData(handle, graph, (void*)result, 0));
	std::cout << "FINISHED NVGRAPH_SSSP()\n";
}

//Wrapper for nvgraphConvertTopology(). This allocates memory for the indices and offsets of the new topology. REMEMBER TO FREE THEM!
void nvgraph_convert_topology(nvgraphHandle_t handle, int nedges, int nvertices, void *source_indices_h, void *destination_indices_h, void *edge_data_h, nvgraphTopologyType_t dst_type, void *dst_topology) {
	cudaDataType_t data_type = CUDA_R_32F;
	//Initialize unconverted topology
	nvgraphCOOTopology32I_t current_topology = (nvgraphCOOTopology32I_t)malloc(sizeof(struct nvgraphCOOTopology32I_st));
	current_topology->nedges = nedges;
	current_topology->nvertices = nvertices;
	current_topology->tag = NVGRAPH_UNSORTED; //NVGRAPH_UNSORTED, NVGRAPH_SORTED_BY_SOURCE or NVGRAPH_SORTED_BY_DESTINATION can also be used
	cudaMalloc((void**)&(current_topology->destination_indices), nedges*sizeof(int));
	cudaMalloc((void**)&(current_topology->source_indices), nedges*sizeof(int));

	//Copy data into topology
	cudaMemcpy(current_topology->destination_indices, destination_indices_h, nedges*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(current_topology->source_indices, source_indices_h, nedges*sizeof(int), cudaMemcpyHostToDevice);

	//Allocate and copy edge data
	float *edge_data_d, *dst_edge_data_d;
	cudaMalloc((void**)&edge_data_d, nedges*sizeof(float));
	cudaMalloc((void**)&dst_edge_data_d, nedges*sizeof(float));
	cudaMemcpy(edge_data_d, edge_data_h, nedges*sizeof(float), cudaMemcpyHostToDevice);

	int *indices_h, *offsets_h, **indices_d, **offsets_d;
	//These are needed for compiler issues (the possibility that the initialization is skipped)
	nvgraphCSCTopology32I_t csc_topology;
	nvgraphCSRTopology32I_t csr_topology;
	switch (dst_type) {
	case NVGRAPH_CSC_32:
		csc_topology = (nvgraphCSCTopology32I_t)dst_topology;
		indices_d = &(csc_topology->source_indices);
		offsets_d = &(csc_topology->destination_offsets);
		break;
	case NVGRAPH_CSR_32:
		csr_topology = (nvgraphCSRTopology32I_t)dst_topology;
		indices_d = &(csr_topology->destination_indices);
		offsets_d = &(csr_topology->source_offsets);
		break;
	default:
		std::cout << dst_type << " is not supported!\n";
		system("PAUSE");
		exit(0);
	}
	cudaMalloc((void**)(indices_d), nedges*sizeof(int));
	cudaMalloc((void**)(offsets_d), (nvertices + 1)*sizeof(int));
	indices_h = (int*)malloc(nedges*sizeof(int));
	offsets_h = (int*)malloc((nvertices + 1)*sizeof(int));

	check(nvgraphConvertTopology(handle, NVGRAPH_COO_32, current_topology, edge_data_d, &data_type, dst_type, dst_topology, dst_edge_data_d));

	//Copy converted topology from device to host
	cudaMemcpy(indices_h, *indices_d, nedges*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(offsets_h, *offsets_d, (nvertices + 1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_data_h, dst_edge_data_d, nedges*sizeof(float), cudaMemcpyDeviceToHost);

	//Free allocated device memory
	cudaFree(indices_d);
	cudaFree(offsets_d);
	cudaFree(edge_data_d);
	cudaFree(dst_edge_data_d);
	cudaFree(current_topology->destination_indices);
	cudaFree(current_topology->source_indices);
	free(current_topology);

	//Set the indices and offsets of dst_topology to the host memory
	switch (dst_type) {
	case NVGRAPH_CSC_32:
		csc_topology->destination_offsets = offsets_h;
		csc_topology->source_indices = indices_h;
		break;
	case NVGRAPH_CSR_32:
		csr_topology->destination_indices = indices_h;
		csr_topology->source_offsets = offsets_h;
		break;
	default:
		std::cout << dst_type << " is not supported!\n";
		system("PAUSE");
		exit(0);
	}
}

// Reads in a file with the format "source\tdestination" where any lines that begin with # are skipped
void load_from_file(std::string file_name, int* source, int *destination, float *edge_data, int nedges) {
	std::ifstream graphData(file_name);
	if (!graphData.is_open()) {
		printf("File failed to open!\n");
		system("PAUSE");
		exit(0);
	}

	int edge_no = 0;
	std::string line;
	if (DEBUG)
		std::cout << "Reading in " << file_name << "...\n";
	while (std::getline(graphData, line)) {
		if (line[0] == '#') { continue; }

		std::stringstream iss(line);
		iss >> source[edge_no] >> destination[edge_no] >> edge_data[edge_no]; //this can be modified to allow edge data to be acquired from the file i.e.: source\tdestination\tedge_weight
																			  //edge_data[edge_no] = 1.0; //This is only being used for file "web-Google.txt"
		edge_no++;
	}
	if (DEBUG)
		std::cout << "Finished!\n";
	graphData.close();
}