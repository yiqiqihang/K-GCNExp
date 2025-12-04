import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import mpl_toolkits.axisartist as axisartist
from mpl_toolkits.axisartist.axislines import Axes
time_scale_factor = [1,1,1,1,2,2,2,4,4,4]

def stgcn_visualize_3d(pose,
                    edge,
                    feature,
                    label=None,
                    label_sequence=None,
                    height=1080,
                    fps=None,
                    activate_2d=True):

    _, T, V, M = pose.shape
    feature = feature[0]
    pos_track = [None] * M

    x_min, x_max = pose[0,:,:,:].min(), pose[0,:,:,:].max()
    y_min, y_max = pose[1,:,:,:].min(), pose[1,:,:,:].max()
    z_min, z_max = pose[2,:,:,:].min(), pose[2,:,:,:].max()

    for t in range(T):

        fig = plt.figure(figsize=(2, 2))
        if activate_2d:
            ax = fig.add_subplot(111, axes_class=Axes)
        else:
            ax = fig.add_subplot(111, projection='3d',axes_class=Axes)  #这种方法也可以画多个子图
        # bx = fig.add_subplot(122)

        for m in range(M):

            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                zi = pose[2, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                zj = pose[2, t, j, m]
                if activate_2d:
                    ax.plot([xi, xj], [yi, yj], 'gray')
                else:
                    ax.plot3D([xi, xj], [yi, yj], [zi, zj], 'gray')
                    # 3D plot also need view_init() to transform the view

        # generate mask
        feature = np.abs(feature)
        # feature = feature / feature.mean()
        max_feature = feature.max()
        for m in range(M):

            f = feature[t, :]
            # f = ((f - f.min()) / (f.max() - f.min()) + 1) ** 5 + 10
            f = ((f - f.min()) / (f.max() - f.min()))
            for v in range(V):
                x = pose[0, t, v, m]
                y = pose[1, t, v, m]
                z = pose[2, t, v, m]

                if activate_2d:
                    # ax.scatter(x,y,color='blue',s=f[v], alpha=0.5)
                    ax.scatter(x,y,color='red',s=20, alpha=f[v])
                    ax.axis('off')
                else:
                    ax.scatter3D(x,y,z, cmap='Blues')

        # bx.barh(range(1, 26), feature[t, :], alpha=0.6)
        # plt.ylim(0.5, 25.5)
        # plt.xlim(0, max_feature)

        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # x_ticks = np.arange(x_min-0.2, x_max+0.2, 0.1)
        # y_ticks = np.arange(y_min-0.2, y_max+0.2, 0.1)

        if not activate_2d:
            ax.set_zlabel('z')
            z_ticks = np.arange(z_min-0.2, z_max+0.2, 0.1)

        

        # 此方法速度慢
        # fig.canvas.draw()

        # # Now we can save it to a numpy array.
        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # 此方法将数据存入缓存，速度快但是耗内存
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        plt.close()

        yield img

def plot_action(pose, edge, feature, save_dir, save_type):

    _, T, V, M = pose.shape
    feature = np.array(feature[0])
    fig = plt.figure(figsize=(10, 10))
    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    z_max = 0
    
    fea = ((feature - feature[::5,:,:].min()) / (feature[::5,:,:].max() - feature[::5,:,:].min() + 1e-12))
    
    for t in range(0,T,5):

        if t: 
            z_max += pose[2, t-1, :, :].max() * 1.5
        f = fea[t, :]
        # print(fea.size())

        for m in range(M):
            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                if not t:
                    zi = pose[2, t, i, m]
                    zj = pose[2, t, j, m]
                else:
                    zi = pose[2, t, i, m] + z_max
                    zj = pose[2, t, j, m] + z_max

                # ax.plot3D([xi, xj], [yi, yj], [zi, zj], 'gray')
                ax.plot3D([zi, zj], [xi, xj], [yi, yj], 'Blue', alpha=0.6)

            for v in range(V):
                x = pose[0, t, v, m]
                y = pose[1, t, v, m]
                z = pose[2, t, v, m] + z_max
                ax.scatter3D(z,x,y, color='red', s=20, alpha=f[v, m])
    font = {'family': 'Times New Roman','weight': 'normal', 'size' : 23,}
    # x_min, x_max = pose[0,:,:,:].min(), pose[0,:,:,:].max()
    # y_min, y_max = pose[1,:,:,:].min(), pose[1,:,:,:].max()
    # z_min, z_max = pose[2,:,:,:].min(), z_max + pose[2, -1 ,:,:].max()
    # ax.set_xlabel(' time\n----->', fontsize=20)
    # ax.xaxis.labelpad = 16
    ax.set_xlabel('time')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # x_ticks = np.arange(x_min-0.1, x_max+0.1, (x_max - x_min)/5)
    # y_ticks = np.arange(y_min-0.2, y_max+0.2, (y_max - y_min)/5)
    # z_ticks = np.arange(z_min-0.2, z_max+0.2, (z_max - z_min)/5)
    # ax.set_xticks(z_ticks)
    # ax.set_yticks(x_ticks)
    # ax.set_zticks(y_ticks)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # ax.view_init(elev=0, azim=0)
    ax.set_box_aspect((20,6,6))

    print("saveing")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('{}/{}.svg'.format(save_dir, save_type))

    return 
    

def stgcn_visualize_hidden(pose,
                    edge,
                    feature,
                    label=None,
                    voting_proba=None,
                    label_sequence=None,
                    video_proba=None,
                    height=1080,
                    fps=None,
                    activate_2d=True):

    _, T, V, M = pose.shape

    for t in range(T):

        fig = plt.figure(figsize=(4*4, 4*5))

        title_str = 'Sequence Predict: {0:.2f}%    {1}\nFrame Predict: {2:.2f}%    {3}\n'.format(
            voting_proba * 100, label, video_proba[t//4][0] * 100, label_sequence[t//4][0])
        plt.title(title_str, fontsize=20)
        plt.axis('off') 

        if activate_2d:
            ax = [fig.add_subplot(5, 4, 2*i+1) for i in range(10)]
        else:
            ax = [fig.add_subplot(121, projection='3d') for i in range(10)]  #这种方法也可以画多个子图
        bx = [fig.add_subplot(5, 4, 2*i) for i in range(1, 11)]

        for m in range(M):

            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                zi = pose[2, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                zj = pose[2, t, j, m]
                if activate_2d:
                    for k in range(10):
                        ax[k].plot([xi, xj], [yi, yj], 'gray')
                else:
                    for k in range(10):
                        ax[k].plot3D([xi, xj], [yi, yj], [zi, zj], 'gray')

        # generate mask
        feature = [np.abs(f) for f in feature]
        # feature = feature / feature.mean()
        max_feature = [f.max() for f in feature]

        for m in range(M):
            score = pose[2, t, :, m].max()
            if score < 0.3:
                continue

            # f = feature[t // 4, :, m]**5
            # if f.mean() != 0:
            #     f = f / f.mean()
            for i in range(10):
                ts = time_scale_factor[i]
                f = feature[i][t // ts, :, m]
                f = ((f - f.min()) / (f.max() - f.min()) + 2) ** 5 + 10

                for v in range(V):
                    x = pose[0, t, v, m]
                    y = pose[1, t, v, m]
                    z = pose[2, t, v, m]

                    ax[i].set_xlabel('x')
                    ax[i].set_ylabel('y')
                    if activate_2d:
                        ax[i].scatter(x,y,color='blue',s=f[v], alpha=0.5)
                        ax[i].annotate(str(v),(x,y))
                        ax[i].axis('off')
                    else:
                        ax[i].scatter3D(x,y,z, cmap='Blues')
                        ax[i].set_zlabel('z')

        # 当前仅支持单人
        for i in range(10):
            ts = time_scale_factor[i]
            bx[i].barh(range(1, 26), feature[i][t // ts, :, 0], alpha=0.6)
            plt.ylim(0.5, 25.5)
            plt.xlim(0, max_feature[i])
        

        # 此方法速度慢
        # fig.canvas.draw()

        # # Now we can save it to a numpy array.
        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # 此方法将数据存入缓存，速度快但是耗内存
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        plt.close()
        
        if t % 5 == 0:
            print("Output frame {} finish.".format(t))
        yield img

def put_text(img, text, position, scale_factor=1):
    t_w, t_h = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_TRIPLEX, scale_factor, thickness=1)[0]
    H, W, _ = img.shape
    position = (int(W * position[1] - t_w * 0.5),
                int(H * position[0] - t_h * 0.5))
    params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
              (255, 255, 255))
    cv2.putText(img, text, *params)


def blend(background, foreground, dx=20, dy=10, fy=0.7):

    foreground = cv2.resize(foreground, (0, 0), fx=fy, fy=fy)
    h, w = foreground.shape[:2]
    b, g, r, a = cv2.split(foreground)
    mask = np.dstack((a, a, a))
    rgb = np.dstack((b, g, r))

    canvas = background[-h-dy:-dy, dx:w+dx]
    imask = mask > 0
    canvas[imask] = rgb[imask]

def extract_correlation(feature):

    # Calculate node features correlation in diff frame and diff layers
    # Using cosin distance

    # np.einsum('ctvm,ctvm->tvm',)
    _, T, __, M = feature[0].size()
    corr_map = np.zeros((len(feature), T, 25, 25, M), dtype=np.float)
    for layer in range(len(feature)):
        C, T, V, M = feature[layer].size()
        for t in range(T):
            for m in range(M):
                f = feature[layer][:, t, :, m].cpu().detach().numpy()
                length = ((f*f).sum(axis=0)**0.5)
                # print(np.outer(length, length))
                corr_map[layer, t, :, :, m] = np.dot(f.T, f)/np.outer(length, length)
    
    for t in range(T):
        fig = plt.figure(figsize=(6*5, 5.5*2))
        for layer in range(len(feature)):
            ax = fig.add_subplot(2, 5, layer + 1)
            ax = sns.heatmap(corr_map[layer, t, :, :, 0])

        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        plt.close()
        print("time:", t)
        yield img