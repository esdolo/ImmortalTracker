import matplotlib.pyplot as plt, numpy as np
from ..data_protos import BBox


class Visualizer2D:
    def __init__(self, name=''):
        plt.rcParams['axes.facecolor'] = 'white'
        self.figure = plt.figure(name,figsize=(14, 14))
        plt.axis('equal')
        self.COLOR_MAP = {
            'white': np.array([255, 255, 255]) / 256,
            'gray': np.array([140, 140, 136]) / 256,
            'light_blue': np.array([4, 157, 217]) / 256,
            'dark_blue': np.array([4, 100, 140]) / 256,
            'red': np.array([191, 4, 54]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'purple': np.array([224, 133, 250]) / 256, 
            'dark_green': np.array([32, 64, 40]) / 256,
            #'green': np.array([77, 115, 67]) / 256
            'green': np.array([77, 211, 67]) / 256
        }
        #self.figure.set_facecolor('black')
    
    def show(self):
        plt.show()
    
    def close(self):
        plt.close()
    
    def save(self, path):
        plt.savefig(path)
    
    def handler_pc(self, pc, color='gray'):
        vis_pc = np.asarray(pc)
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], marker='o', color=color, s=0.1)
    
    def handler_box(self, box: BBox, message: str='', color='red', linestyle='solid',linewidth=0.1):
        corners = np.array(BBox.box2corners2d(box))[:, :2]
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=color, linestyle=linestyle)
        #plt.scatter(np.mean(corners[:, 0]), np.mean(corners[:, 1]), marker='o', color=self.COLOR_MAP[color], s=1)
        corner_index = 0#np.random.randint(0, 4, 1)
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=color)
        self.figure.set_facecolor('black')

    def handler_box_centerpoint(self, box: BBox, message: str='', color='red', linestyle='solid',s=10):
        corners = np.array(BBox.box2corners2d(box))[:, :2]
        corners = np.concatenate([corners, corners[0:1, :2]])
        #plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)
        plt.scatter(np.mean(corners[:, 0]), np.mean(corners[:, 1]), marker='o', color=color, s=s)
        # corner_index = np.random.randint(0, 4, 1)
        # plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[color])
        #self.figure.set_facecolor('black')
        