# script should be run from root of project, paths are written relative to root
vgg16_ini=./data/configs/VSD/VSD_VGG16*ini

for file in ${vgg16_ini};do
	searchnets train ${file}
	searchnets test ${file}
done