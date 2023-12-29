import numpy as np
import matplotlib.pyplot as plt
import shap
from matplotlib import gridspec

class Ploting:
    def __init__(self,plot_type='shap',shap_values='None',feature_values='None',header_list='None'):
        if plot_type=='shap':
            self.shap_values = shap_values
            self.feature_values = feature_values
            self.header_list = header_list

    
    def shap_bee_img(self,path,plot_size = (16,16),max_display = 10):
        
        shap_values = self.shap_values
        feature_values = self.feature_values
        header_list = self.header_list
        plt.rc('axes', linewidth=2)
        shap.summary_plot(shap_values, features=feature_values, feature_names=header_list, max_display=max_display,
                        plot_size=plot_size,show=False,)

        plt.xlim((-5,5))

        plt.gcf().axes[-1].set_aspect(100)
        plt.gcf().axes[-1].set_box_aspect(100)
        fig, ax = plt.gcf(), plt.gca()
        plt.rc('axes', linewidth=2)

        ax.tick_params(labelsize=40,colors='black')
        ax.set_xlabel("SHAP Values",fontsize=40)
        ax.spines['bottom'].set_position(('axes', 0.015))
        cb_ax = fig.axes[1]


        cb_ax.tick_params(labelsize=40)
        cb_ax.set_ylabel("Feature value",fontsize=40,labelpad=-50)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def shap_bar_img(self,image_num, path,plot_size = (16,16),max_display = 10):
        shap_values = self.shap_values
        feature_values = self.feature_values
        header_list = self.header_list
        plt.rcParams.update({'font.size': 40})


        plt.rc('axes', linewidth=2)
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])

        fig, ax = plt.subplots(figsize=plot_size)
        abs_shap_values = np.abs(shap_values)
        mean_abs = np.mean(abs_shap_values,axis=0)
        std_abs = np.std(abs_shap_values,axis=0)
        pool = list(zip(list(mean_abs),list(std_abs),header_list,list(range(len(header_list)))))
        pool.sort()
        print(pool)
        pool = pool[-max_display:]

        plt.rc('axes', linewidth=2)

        for a in pool:
            if a[3]<image_num:
                plt.barh([a[2]],[a[0]],height=0.6,color='#2889D1')#'#63079D')##'6C38E5')
            else:
                plt.barh([a[2]],[a[0]],height=0.6,color='#CC3951')#'#e5383b')
        #mean_shap_list = np.mean(abs_shap_list,axis=1).transpose(1,0)
        ax.set_xlabel('Mean(|SHAP|)')#, fontsize=12)
        ax.tick_params('y',length=20, width=0.5, which='major')
        ax.set_ylim((-1,min(max_display,len(pool))))
        ax.set_xticks([0.00, 0.05, 0.10, 0.15])

        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_bounds((-0.3,min(max_display,len(pool))-0.6))

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    
    def shap_bar(self):        
        shap_arr = self.shap_values
        feature_arr = self.feature_values
        feature_name = self.header_list
        fontsize = 60
        plt.rcParams.update({'font.size': fontsize})
        plt.figure(figsize=(15,12))
        gs = gridspec.GridSpec(9, 7)
        plt.subplot(gs[0:8, 3:7])
        shap.bar_plot(shap_arr,feature_arr,feature_name,max_display=5,show=False)
        #print(feature_arr)
        plt.rcParams.update({'font.size': fontsize})

        pool = list(zip(list(np.abs(shap_arr)),feature_name))
        pool.sort()
        pool = pool[-5:]
        plt.yticks(list(range(1,6)),[a[1] for a in pool],fontsize=fontsize)
        plt.xlabel("SHAP Values")
        plt.gca().spines['bottom'].set_linewidth(5)
        plt.gca().spines['left'].set_linewidth(5)
        gca = plt.gca()
        print([i for i in gca.containers])
        ilist = list(range(5))
        ilist.remove(0)
        for i in ilist:
            print(i, end=' ')
            gca.containers[0][i].set_color('silver')

p = Ploting(
    shap_values=np.load('results/multimodal_shap_values.npy'),
    feature_values=np.load('results/multimodal_feature_values.npy'),
    header_list=['CTA(Image)', 'CTA+(Image)', 'NCCT(Image)', 'CBV(Image)', 'CBF(Image)', 'Tmax(Image)', 'MTT(Image)','Baseline NIHSS', 'Premorbid mRS', 'Gender', 'Age',
                   'BMI', 'Hypertension','Smoking', 'Drinking', 'Diabetes', 'Hyperlipidemia', 'Atrial Fibrillation',
                   'Thrombolysis','Ant/Post Circ']
)

p.shap_bee_img('results/bee')
p.shap_bar_img(7,'results/bar')