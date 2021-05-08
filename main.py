import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as ip
import matplotlib.mlab as ml
from matplotlib.colors import LightSource
from matplotlib import cm
import xml.dom.minidom

def degree2rad(degree):
	return degree * np.pi / 180

def distanceRatio2LL(latMIN, latMAX, lonMIN, lonMAX):
	latMIN_rad = degree2rad(latMIN)
	latMAX_rad = degree2rad(latMAX)
	lonMIN_rad = degree2rad(lonMIN)
	lonMAX_rad = degree2rad(lonMAX)

	''' deterministic longitude '''
	a1 = latMAX_rad - latMIN_rad
	b1 = 0
	distance1 = 2 * np.arcsin(np.sqrt(np.sin(a1/2)**2 + np.cos(latMAX_rad)*np.cos(latMIN_rad)*np.sin(b1/2)**2)) * 6378.137
	dis2lat = distance1 / (latMAX - latMIN) # km/degree

	''' deterministic latitude '''
	a2 = 0
	b2 = lonMAX_rad - lonMIN_rad
	distance2 = 2 * np.arcsin(np.sqrt(np.sin(a2 / 2) ** 2 + np.cos(latMAX_rad) * np.cos(latMIN_rad) * np.sin(b2 / 2) ** 2)) * 6378.137
	dis2lon = distance2 / (lonMAX - lonMIN) # km/degree

	return dis2lat, dis2lon

def gridCheck(grid_index_x_len, grid_index_y_len, grid_index_z_len,
              points_LonLat, terrian_height_2meter,
              buildingORblock_information, gridSize):
	grid_existence_matrix = np.ones((grid_index_x_len, grid_index_y_len, grid_index_z_len), dtype=int)
	''' terrian inforamtion '''
	for grid_index_x in range(grid_index_x_len):
		for grid_index_y in range(grid_index_y_len):
			x_center = (grid_index_x+0.5) * gridSize
			y_center = (grid_index_y+0.5) * gridSize
			x_y_center = np.array((x_center, y_center)).reshape((1, 2))
			terrian_height_x_y_center = ip.griddata(points_LonLat, terrian_height_2meter, x_y_center, method='nearest')
			index_limit =  int(np.round(terrian_height_x_y_center / gridSize))
			if index_limit:
				grid_existence_matrix[grid_index_x, grid_index_y, 0:index_limit] = 0
				# for index_z in range(index_limit):
				# 	grid_existence_matrix[grid_index_x, grid_index_y, index_z] = 0
	''' building information'''
	for building_information in buildingORblock_information:
		building_center = building_information[0]
		building_height_top = building_information[1]
		x_building_center = building_center[0]
		y_building_center = building_center[1]

		x_buffer_min = int((x_building_center + 100) //gridSize)
		x_buffer_max = math.ceil((x_building_center+100)/gridSize)

		y_buffer_min = int((y_building_center+100) // gridSize)
		y_buffer_max = math.ceil((y_building_center+100) / gridSize)

		z_buffer_max = math.ceil(building_height_top / gridSize)
		grid_existence_matrix[x_buffer_min:x_buffer_max,y_buffer_min:y_buffer_max,0:z_buffer_max] = 0
		# for x in range(x_buffer_min, x_buffer_max):
		# 	for y in range(y_buffer_min, y_buffer_max):
		# 		for z in range(z_buffer_max):
		# 			grid_existence_matrix[x, y, z] = 0

	return grid_existence_matrix

def main(file_terrian, file_geometry, numDivided, gridSize):
	# region terrian information
	terrian_data_initial = pd.read_csv(file_terrian) # longitute, latitute, height
	dataSize = terrian_data_initial.shape[0]
	terrian_longitude_initial = np.array(terrian_data_initial['Lon/X']).reshape((dataSize,1))
	terrian_latitude_initial = np.array(terrian_data_initial['Lat/Y']).reshape((dataSize,1))
	terrian_height_initial = np.array(terrian_data_initial['Hei/H']).reshape((dataSize,1))

	lonMIN = terrian_longitude_initial.min()
	lonMAX = terrian_longitude_initial.max()
	latMIN = terrian_latitude_initial.min()
	latMAX = terrian_latitude_initial.max()
	dis2lat, dis2lon = distanceRatio2LL(latMIN, latMAX, lonMIN, lonMAX)

	terrian_longitude_2meters = (terrian_longitude_initial - lonMIN) * dis2lon * 1000 # [0, XXX]
	terrian_latitude_2meters = (terrian_latitude_initial - latMIN) * dis2lat * 1000 # [0, XXX]
	terrian_height_2meter = terrian_height_initial - terrian_height_initial.min() # [0, XXX]

	longitude_longitude = np.reshape(terrian_longitude_2meters, (len(terrian_longitude_2meters)//numDivided, numDivided))
	latitude_latitude = np.reshape(terrian_latitude_2meters, (len(terrian_latitude_2meters)//numDivided, numDivided))
	height_height = np.reshape(terrian_height_2meter, (len(terrian_height_2meter)//numDivided, numDivided))

	fig = plt.figure(figsize=(10,10))
	# ax3D = plt.axes(projection='3d')
	ax3D = fig.add_subplot(projection = '3d')
	ax3D.invert_xaxis()
	ax3D.plot_surface(longitude_longitude, latitude_latitude, height_height, rstride=1, cstride=1, cmap=cm.Spectral, antialiased=False, shade=False)
	# plt.show()
	# endregion

	# region gemeotry information
	geomeotry_data_tree = xml.dom.minidom.parse(file_geometry)
	geomeotry_data_collection = geomeotry_data_tree.documentElement
	''' node information '''
	# node information: id, lat, lon
	nodes_collection = geomeotry_data_collection.getElementsByTagName("node")
	nodeIndex = []
	nodesPosition = []
	points_LonLat = np.hstack((terrian_longitude_2meters, terrian_latitude_2meters))
	for node in nodes_collection:
		index = node.getAttribute('id')
		Lat2meter = (np.float(node.getAttribute('lat')) - latMIN) * dis2lat * 1000
		Lon2meter = (np.float(node.getAttribute('lon')) - lonMIN) * dis2lon * 1000
		if node.getAttribute('visible') == 'true':
			nodeIndex.append(index)
			nodesPosition.append(np.array((Lon2meter, Lat2meter)))
	nodesPosition =  np.array(nodesPosition)
	nodeHei = ip.griddata(points_LonLat, terrian_height_2meter, nodesPosition, method='nearest')
	nodePositionandHei = np.hstack((nodesPosition, nodeHei))
	nodeInformation = pd.DataFrame(index=nodeIndex, columns=['Lon2meter', 'Lat2meter', 'Hei'], data=nodePositionandHei)

	''' way information '''
	# way information: id, nd --> ref,
	# tag k = building, v = yes
	# tag k = height, v = XX --> only partial buildings has the height information
	# tag k = landuse, v= farmyard, commercial, residential, grass, recreation_ground, industrial,
	# forest, village_green, cemetery, construction, cemetery, greenhouse_horticulture, meadow, village_green, landfill
	# tag k = natural, v= water
	# tag k = highway, barrier, parking, waterway, boundary, aeroway, sport
	ways_collection = geomeotry_data_collection.getElementsByTagName("way")
	buildingORblock_information = []
	for way in ways_collection:
		if way.getAttribute('visible') == 'true':
			# required nodes in this way
			nd_ref = way.getElementsByTagName("nd")
			nd_id_in_way = []
			for nd_ref_i in nd_ref:
				nd_id = nd_ref_i.getAttribute('ref')
				nd_id_in_way.append(nd_id)
			nd_information_way = nodeInformation.loc[nd_id_in_way]

			# other information
			tag_info = way.getElementsByTagName("tag")
			way_type = []
			way_height = 0
			for tag in tag_info:
				if tag.getAttribute("k") in ['building', 'Building', 'aeroway', 'highway', 'barrier', 'parking', 'waterway', 'sport', 'boundary']:
					way_type = tag.getAttribute("k")
				elif tag.getAttribute("k") in ['landuse', 'natural']:
					way_type = tag.getAttribute("v")
				elif tag.getAttribute("k") == 'height':
					way_height = np.int(tag.getAttribute("v"))
			# plot the way: node information, height infromation
			way_x_lon2meter = np.array(nd_information_way['Lon2meter'])
			way_y_lat2meter = np.array(nd_information_way['Lat2meter'])
			way_z_hei2meter = np.array(nd_information_way['Hei'])
			''' plot the geometry baseline '''
			# kwargs = {'alpha': 1, 'color': 'black'}
			# ax3D.plot3D(way_x_lon2meter, way_y_lat2meter, way_z_hei2meter, **kwargs)
			# plt.show()
			if way_type in ['building', 'Building']:
				if way_height == 0:
					way_height = 20
				''' plot the 3D volume '''
				# print('lon min in way - lon min in terrian = ', way_x_lon2meter.min() - terrian_longitude_2meters.min())
				# print('lon max in way - lon max in terrian = ', way_x_lon2meter.max() - terrian_longitude_2meters.max())
				# print('lat min in way - lat min in terrian = ', way_y_lat2meter.min() - terrian_latitude_2meters.min())
				# print('lat max in way - lat max in terrian = ', way_y_lat2meter.max() - terrian_latitude_2meters.max())
				# print('/n')
				way_z_hei2meter_top = way_z_hei2meter.min() + way_height

				building_center = [way_x_lon2meter.mean(), way_y_lat2meter.mean(), 0.5 * (way_z_hei2meter + way_z_hei2meter_top).mean()]
				building_top_height = way_z_hei2meter_top
				buildingORblock_information.append([building_center, building_top_height])

				for index in range(len(way_x_lon2meter) - 1):
					point1_x, point1_y, point1_z = way_x_lon2meter[index], way_y_lat2meter[index], way_z_hei2meter[index]
					point2_x, point2_y, point2_z = way_x_lon2meter[index+1], way_y_lat2meter[index+1], way_z_hei2meter[index+1]
					point12_z = np.min((point1_z, point2_z))

					surface_z = np.linspace(point12_z, way_z_hei2meter_top, 10)
					if point2_y == point1_y:
						surface_x = np.linspace(point1_x, point2_x, 10)
						surface_xx, surface_zz = np.meshgrid(surface_x, surface_z)
						surface_yy = point2_y
					elif point1_x == point2_x:
						surface_y = np.linspace(point1_y, point2_y, 10)
						surface_yy, surface_zz = np.meshgrid(surface_y, surface_z)
						surface_xx = point1_x
					else:
						surface_x = np.linspace(point1_x, point2_x, 10)
						surface_xx, surface_zz = np.meshgrid(surface_x, surface_z)
						surface_yy = (point2_y - point1_y) / (point2_x - point1_x) * (surface_xx - point1_x) + point1_y

					ax3D.plot_surface(surface_xx, surface_yy, surface_zz, color='grey')

	''' grid informaiton '''
	grid_index_x_len = math.ceil(terrian_longitude_2meters.max() / gridSize)
	grid_index_y_len = math.ceil(terrian_latitude_2meters.max() / gridSize)
	grid_index_z_len = math.ceil(300 / gridSize)
	grid_existence_matrix = gridCheck(grid_index_x_len, grid_index_y_len, grid_index_z_len,
	                                  points_LonLat, terrian_height_2meter,
	                                  buildingORblock_information, gridSize)
	zeros_index = np.argwhere(grid_existence_matrix == 0)
	for grid_index_x in range(grid_index_x_len):
		for grid_index_y in range(grid_index_y_len):
			for grid_index_z in range(grid_index_z_len):
				if grid_existence_matrix[grid_index_x, grid_index_y, grid_index_z] == 1:
					gird_index = np.array([grid_index_x, grid_index_y, grid_index_z])
					Indexpoint_position = gird_index * gridSize
					grid_center_position = Indexpoint_position + 0.5 * gridSize
					x, y, z = Indexpoint_position[0], Indexpoint_position[1], Indexpoint_position[2]
					xx = [x, x, x + gridSize, x + gridSize, x]
					yy = [y, y + gridSize, y + gridSize, y, y]
					kwargs = {'alpha': 1, 'color': 'blue'}
					ax3D.plot3D(xx, yy, [z] * 5, **kwargs)
					ax3D.plot3D(xx, yy, [z + gridSize] * 5, **kwargs)
					ax3D.plot3D([x, x], [y, y], [z, z + gridSize], **kwargs)
					ax3D.plot3D([x, x], [y + gridSize, y + gridSize], [z, z + gridSize], **kwargs)
					ax3D.plot3D([x + gridSize, x + gridSize], [y + gridSize, y + gridSize], [z, z + gridSize], **kwargs)
					ax3D.plot3D([x + gridSize, x + gridSize], [y, y], [z, z + gridSize], **kwargs)

	ax3D.set_xlabel('Longitude')
	ax3D.set_ylabel('Latitude')
	ax3D.set_zlabel('Height')
	# ax3D.set_xlim(0, 20 *  math.ceil(longitude_longitude.max() / 20))
	# ax3D.set_ylim(0, 20 * math.ceil(latitude_latitude.max() / 20))
	# ax3D.set_zlim(0, 300)
	plt.show()
	a = 1
	# endregion

	return True


if __name__ == "__main__":
	gridSize = 20
	accuracy = '20'  # 5, 8, 10
	file_geometry = "map_university.osm"
	file_terrian = "elevation_university_" + accuracy + "m.csv"

	if accuracy == '10':
		numDivided = 203
	elif accuracy == '8':
		numDivided = 557
	elif accuracy == '5':
		numDivided = 1036
	elif accuracy == '20':
		numDivided = 149

	main(file_terrian, file_geometry, numDivided, gridSize)