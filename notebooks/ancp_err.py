import os
import ancpbids
from ancpbids.query import query_entities

#dataset_path = '/Volumes/SSD_DATA/MEG_data/openneuro/ds003483'
dataset_path = '/Volumes/SSD_DATA/MEG_data/openneuro/ds000117'
#html_str = '<!DOCTYPE html><html><head><title>Page Title</title></head><body><h1>This is a Heading</h1><p>This is a paragraph.</p></body></html>'
html_str = 'testing stuff ancp'


dataset = ancpbids.load_dataset(dataset_path)
schema = dataset.get_schema()

#create derivatives folder first:
derivatives_path = os.path.join(dataset_path, 'derivatives')
if not os.path.isdir(derivatives_path):
    os.mkdir(derivatives_path)

derivative = dataset.create_derivative(name="Meg_QC_test")
derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"


entitiesR = query_entities(dataset, scope='raw')
print('___MEGqc___: ', 'entitiesR', entitiesR)
entitiesD = query_entities(dataset, scope='derivatives')
print('___MEGqc___: ', 'entitiesD', entitiesD)


test_folder = derivative.create_folder(name='calculation')

meg_artifact = test_folder.create_artifact()

meg_artifact.add_entity('desc', 'TestHtmlFile') 
meg_artifact.suffix = 'meg'
meg_artifact.extension = '.html'

meg_artifact.content = lambda file_path, cont=html_str: (
    open(file_path, 'w').write(cont)
)

print('___MEGqc___: ',  'NOW WE PREPARAED ARTIFACT')
entitiesR = query_entities(dataset, scope='raw')
print('___MEGqc___: ', 'entitiesR', entitiesR)
entitiesD = query_entities(dataset, scope='derivatives')
print('___MEGqc___: ', 'entitiesD', entitiesD)

ancpbids.write_derivative(dataset, derivative) 

print('___MEGqc___: ', 'DONE')