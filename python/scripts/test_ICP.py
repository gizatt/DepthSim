import os,sys
sys.path.insert(0, '../')

import numpy as np
from scipy import misc
from common import common
from common import util
from director import vtkAll as vtk
import yaml
import time
from director import vtkNumpy as vnp
from director import filterUtils
from RGBDCNN import network
import matplotlib.pyplot as plt
from vtk.util import numpy_support

def set_shader_input(mapper):
  mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,  #// in the fragment shader
    "//VTK::Light::Impl", #// replace the normal block
    True, #// before the standard replacements
    "fragOutput0 = vec4((normalVCVSOutput.x+1)/2.0,(normalVCVSOutput.y+1)/2.0,(normalVCVSOutput.z+1)/2.0, 1);\n",
    True #// only do it once
    )

def set_shader(mapper):
	mapper.SetVertexShaderCode(
    "//VTK::System::Dec\n"  
    "attribute vec4 vertexMC;\n"
    "attribute vec3 normalMC;\n"
    "uniform mat3 normalMatrix;\n"
    "uniform mat4 MCDCMatrix;\n"
    "uniform mat4 MCVCMatrix;\n"
    "varying vec3 normalVCVSOutput;\n"
    "varying vec4 vertexVCVSOutput;\n"
    "attribute vec2 tcoordMC;\n"
    "varying vec2 tcoordVCVSOutput;\n"
    "void main () {\n"
    "  normalVCVSOutput = normalMatrix * normalMC;\n"
    "  tcoordVCVSOutput = tcoordMC;\n"
    "  vertexVCVSOutput = MCVCMatrix * vertexMC;\n"
    "  gl_Position = MCDCMatrix * vertexMC;\n"
    "}\n")

	mapper.SetFragmentShaderCode(
    "//VTK::System::Dec\n"  
    "//VTK::Output::Dec\n"  
    "varying vec3 normalVCVSOutput;\n"
    "varying vec4 vertexVCVSOutput;\n"
    "varying vec2 tcoordVCVSOutput;\n"
    "out vec4 colorOut;\n"
    "uniform float z_near;\n"
    "uniform float z_far;\n"
    "uniform sampler2D texture_0;\n"
    "void main () {\n"
    "  float z = -vertexVCVSOutput.z;  // In meters.\n"
    "  float random_noise = texture(texture_0, vec2(gl_FragCoord.x / 640., gl_FragCoord.y / 480.)).x;\n"
    # // "  float f = pow(dot(normalVCVSOutput, normalize(vertexVCVSOutput.xyz)), 5.) / pow(z, 2.);\n"
    # // "  float angle = dot(normalVCVSOutput, normalize(-vertexVCVSOutput.xyz)) / 2. + 0.5;\n"
    # // "  float distance = 1. / pow(z, 2.);\n"
    # // "  float dnoise = distance * (1. + 0.01 * random_noise);\n"
    # // "  float f = angle / pow(z, 2.) + random_noise;\n"
    # // "  float z = gl_FragCoord.z; // [0, 1].\n"
    "  float angle = dot(normalVCVSOutput, normalize(-vertexVCVSOutput.xyz));\n"
    "  if (angle + random_noise * 0.2 > 0.3 * min(1.0, z * 2.5 / 5.)) {\n"
    "    float z_noise = z * (1. + 0.01 * random_noise);\n"
    "    float z_norm = (z_noise - z_near) / (z_far - z_near); // From meters to [0, 1].\n"
    "    colorOut = vec4(z_norm, z_norm, z_norm, 1.);\n"
    "  } else {\n"
    "    colorOut = vec4(0, 0, 0, 1.);\n"
    "  }\n"
    "  float z_norm = (z - z_near) / (z_far - z_near); // From meters to [0, 1].\n"
    "  z_norm = clamp(z_norm, 0.f, 1.f);\n"
    "  colorOut = vec4(z_norm, z_norm, z_norm, 1.);\n"
    "}\n")
 
### enumerate tests try only cluttered scenes
objects_per_scene = 5
path  = "/media/drc/DATA/chris_labelfusion/CORL2017/logs_test/"
paths = []
for f in os.listdir(path):
	if "2017" in f:
		if "registration_result.yaml" in os.listdir(path+f):
			with open(path+f+"/registration_result.yaml") as read:
				transformYaml = yaml.load(read)
				if len(transformYaml.keys()) >=objects_per_scene:
					paths.append((f,transformYaml.keys())) 

print "extracting from scenes..."
for i in paths:
	print i[0]

## setup rendering enviornment for mesh
view_height = 480
view_width = 640
renderer = vtk.vtkRenderer()
renderer.SetViewport(0,0,0.5,1)
renWin = vtk.vtkRenderWindow()
interactor = vtk.vtkRenderWindowInteractor()
renWin.SetSize(2*view_width,view_height)
camera = vtk.vtkCamera()
renderer.SetActiveCamera(camera);
renWin.AddRenderer(renderer);
interactor.SetRenderWindow(renWin);
common.set_up_camera_params(camera)

### setup rendering enviornment for point cloud
renderer1 = vtk.vtkRenderer()
renderer1.SetViewport(0.5,0,1,1)
camera1 = vtk.vtkCamera()
renderer1.SetActiveCamera(camera1);
renWin.AddRenderer(renderer1);
common.set_up_camera_params(camera1)
renSource = vtk.vtkRendererSource()
renSource.SetInput(renderer)
renSource.WholeWindowOff()
renSource.DepthValuesOnlyOn()
renSource.Update()

####setup filters
filter1= vtk.vtkWindowToImageFilter()
scale =vtk.vtkImageShiftScale()
filter1.SetInput(renWin)
filter1.SetMagnification(1)
filter1.SetInputBufferTypeToZBuffer()
windowToColorBuffer = vtk.vtkWindowToImageFilter()
windowToColorBuffer.SetInput(renWin)
windowToColorBuffer.SetInputBufferTypeToRGB()     
scale.SetOutputScalarTypeToUnsignedShort()
scale.SetScale(1000);

out_dir = "/media/drc/DATA/chris_labelfusion/RGBDCNN/"
object_dir = "/media/drc/DATA/chris_labelfusion/object-meshes"

out_file = "/media/drc/DATA/chris_labelfusion/RGBDCNN/stats.yaml"
stats = {}
samples_per_run = 1

###run through scenes
for i,j in paths[1:50]:
  #set file names
  data_dir = path+i
  print data_dir
  data_dir_name =  os.path.basename(os.path.normpath(data_dir))
  object_dir = "/media/drc/DATA/chris_labelfusion/object-meshes"
  mesh ='meshed_scene.ply'

  #####set up mesh
  actor = vtk.vtkActor()
  mapper = vtk.vtkPolyDataMapper()
  #set_shader(mapper)
  fileReader = vtk.vtkPLYReader()
  fileReader.SetFileName(data_dir+"/"+mesh)
  mapper.SetInputConnection(fileReader.GetOutputPort())
  actor.SetMapper(mapper)
  renderer.AddActor(actor)

  #add objects
  objects = common.Objects(data_dir,object_dir)

  poses = common.CameraPoses(data_dir+"/posegraph.posegraph")
  for l in np.random.choice(range(1,500),samples_per_run):
    new_data_dir = '/media/drc/DATA/chris_labelfusion/RGBDCNN/'
    utimeFile = open(data_dir+"/images/"+ str(l).zfill(10) + "_utime.txt", 'r')
    utime = int(utimeFile.read())    
    depthFile = new_data_dir+ i+"/"+str(l).zfill(10) + "_depth.png"
    normalFile = new_data_dir+i+"/"+ str(l).zfill(10) +"_"+i+"_normal_ground_truth.png"
    #update camera transform
    cameraToCameraStart = poses.getCameraPoseAtUTime(utime)
    t = cameraToCameraStart
    common.setCameraTransform(camera, t)
    common.setCameraTransform(camera1, t)
    renWin.Render()
    renSource.Update()

    #update filters
    filter1.Modified()
    filter1.Update()
    windowToColorBuffer.Modified()
    windowToColorBuffer.Update()
    
    #extract depth image
    depthImage = vtk.vtkImageData()
    pts = vtk.vtkPoints()
    ptColors = vtk.vtkUnsignedCharArray()
    vtk.vtkDepthImageUtils.DepthBufferToDepthImage(filter1.GetOutput(), windowToColorBuffer.GetOutput(), camera, depthImage, pts, ptColors)
    scale.SetInputData(depthImage)
    scale.Update()

    #source = np.flip(np.reshape(numpy_support.vtk_to_numpy(renSource.GetOutput().GetPointData().GetScalars()),(480,640)),axis=0)


    #modify this for simulated depth
    source = np.flip(np.reshape(numpy_support.vtk_to_numpy(renSource.GetOutput().GetPointData().GetScalars()),(480,640)),axis=0)
    
    #modify this for simulated depth
    depthsim_source = np.copy(source)

    #modify this for real depth mask
    source_real = np.copy(source)

    #modify this for kunis methods
    source_kuni = np.copy(source)

    #simulate depth image
    model_path = "../models/net_depth_seg_v1.hdf5"
    model = network.load_trained_model(weights_path = model_path)
    threshold = .5
    img_height,img_width = (480,640)
    stack = np.zeros((1,img_height,img_width,1)) 
    vtk_array = scale.GetOutput().GetPointData().GetScalars()
    im = np.flip(numpy_support.vtk_to_numpy(vtk_array).reshape(img_height, 2*img_width)[:,:640]/3500.,axis = 0)
    stack[0,:,:,0] = im
    predicted_prob_map = model.predict_on_batch(stack)
    network.apply_mask(predicted_prob_map,depthsim_source,threshold)
    im_depth_sim_vtk = vnp.numpyToImageData(np.reshape(depthsim_source,(480,640,1)),vtktype=vtk.VTK_FLOAT)

    #simulate real
    real_depth = misc.imread(depthFile)
    source_real[real_depth==0]=0
    real_depth_vtk = vnp.numpyToImageData(np.reshape(source_real,(480,640,1)),vtktype=vtk.VTK_FLOAT)

    #simulate kunis
    normals = util.ratio_from_normal(util.convert_rgb_normal(misc.imread(normalFile)))
    source_kuni[normals<.3]=0
    kuni_depth_vtk = vnp.numpyToImageData(np.reshape(source_kuni,(480,640,1)),vtktype=vtk.VTK_FLOAT)

    #real simulation
    source_sim = vnp.numpyToImageData(np.reshape(source,(480,640,1)),vtktype=vtk.VTK_FLOAT)


    iterate = zip([im_depth_sim_vtk,real_depth_vtk,kuni_depth_vtk,source_sim],["depthsim","realdepth","kunidepth","sim"])
    for img,simtype in iterate:
      objects.loadObjectMeshes("/registration_result.yaml",renderer1,keyword=None)
      pc = vtk.vtkDepthImageToPointCloud()
      pc.SetInputData(img)
      pc.SetCamera(renderer.GetActiveCamera())
      pc.ProduceColorScalarsOn()
      pc.ProduceVertexCellArrayOn()
      pc.CullFarPointsOn()
      pc.CullNearPointsOff()
      pc.Update()

      pcMapper = vtk.vtkPolyDataMapper()
      pcMapper.SetInputData(pc.GetOutput())
      pcMapper.Update()
      pcActor = vtk.vtkActor()
      pcActor.SetMapper(pcMapper);
      renderer1.AddActor(pcActor);
      renWin.Render()

      scene = pcActor.GetMapper().GetInput()
      objects.vtkICP(scene)
      objects.update_poses(renderer1)
      renWin.Render()
      print "dumping data"
      objects.dump_icp_results("/media/drc/DATA/chris_labelfusion/logs_final/test_"+str(i)+"_run_"+str(l)+"_"+simtype+".yaml")
      objects.reset()
      renderer1.RemoveAllViewProps();

  renderer.RemoveAllViewProps();
  renWin.Render();
renWin.Render();
interactor.Start();



def inverse_project(camera,depthImage):

  imageWidth = 640
  imageHeight = 480
  aspectRatio = imageWidth/float(imageHeight);

  depthData = np.zeros((imageHeight,imageWidth))

  cameraToViewport = vnp.getNumpyFromTransform(camera.GetProjectionTransformMatrix(aspectRatio, 0, 1))
  viewportToCamera = vnp.getNumpyFromTransform(camera.GetProjectionTransformMatrix(aspectRatio, 0, 1).inverse())
  worldToCamera = vnp.getNumpyFromTransform(camera.GetViewTransformMatrix())
  cameraToWorld = vnp.getNumpyFromTransform(camera.GetViewTransformMatrix().invers())

  for y in range(img_height):
    for x in range(img_width):
      np.array = [(2*float(x)/imageWidth - 1,2*float(y)/imageHeight - 1,depthImage[y,x], 1.0)]
      ptToCamera = viewportToCamera * ptToViewport;
      w3 = 1.0 / ptToCamera[3];

      ptToCamera[0] *= w3;
      ptToCamera[1] *= w3;
      ptToCamera[2] *= w3;
      ptToCamera[3] = 1.0;

      ptToWorld = cameraToWorld * ptToCamera;
      depthData[y,x] = -ptToCamera[2];

  return depthData  

