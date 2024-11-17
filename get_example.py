import numpy as np



for i in range(20):
    img = np.random.rand(7,32,256,256)
    mask = np.random.randint(2,size=(32,256,256))
    np.save('./example/train/image/'+str(i)+'.npy',img)
    np.save('./example/train/mask/'+str(i)+'.npy',mask)


minmax_clinical_data = np.random.rand(20,13)

np.save('./example/train/minmax_clinical_data.npy',minmax_clinical_data)
zscored_clinical_data = (minmax_clinical_data-np.mean(minmax_clinical_data,0))/np.std(minmax_clinical_data)
np.save('./example/train/zscored_clinical_data.npy',zscored_clinical_data)
np.save('./example/train/results.npy',np.random.randint(7,size=(20,1)))

for i in range(5):
    img = np.random.rand(7,32,256,256)
    mask = np.random.randint(2,size=(32,256,256))
    np.save('./example/validate/image/'+str(i)+'.npy',img)
    np.save('./example/validate/mask/'+str(i)+'.npy',mask)


minmax_clinical_data = np.random.rand(5,13)

np.save('./example/validate/minmax_clinical_data.npy',minmax_clinical_data)
zscored_clinical_data = (minmax_clinical_data-np.mean(minmax_clinical_data,0))/np.std(minmax_clinical_data)
np.save('./example/validate/zscored_clinical_data.npy',zscored_clinical_data)

np.save('./example/validate/results.npy',np.random.randint(7,size=(5,1)))