import argparse
import result
from utils.result import ResultDict
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.utils import StopWatch

plt.style.use('ggplot')

parser = argparse.ArgumentParser(description='live plotter for saved result.')
parser.add_argument('--load_dir', type=str)

plot_names = ['SGD', 'RMSprop', 'NAG', 'ADAM', 'LSTM-base']


def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)
        line1 = sns.lineplot(data_frame, x=x, y=y, hue=hue, hue_order=plot_names)
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1

def test():
  size = 5
  x_vec = np.linspace(0,1,size+1)[0:-1]
  y_vec = np.random.randn(len(x_vec))
  line1 = []
  import pdb; pdb.set_trace()
  while True:
      rand_val = np.random.randn(1)
      y_vec[-1] = rand_val
      line1 = live_plotter(x_vec,y_vec,line1)
      y_vec = np.append(y_vec[1:],0.0)


def test2():
  # draw the figure so the animations will work
  fig = plt.gcf()
  fig.show()
  fig.canvas.draw()
  size = 100

  while True:
      # compute something
      x_vec = np.linspace(0,100,size+1)[0:-1]
      y_vec = np.random.randn(len(x_vec))* 100
      plt.plot(x_vec, y_vec)

      # update canvas immediately
      plt.xlim([0, 100])
      plt.ylim([0, 100])
      #plt.pause(0.01)  # I ain't needed!!!
      fig.canvas.draw()

def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

def test3():
  fig1 = plt.figure()

  # Fixing random state for reproducibility
  np.random.seed(19680801)

  data = np.random.rand(2, 25)
  l, = plt.plot([], [], 'r-')
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.xlabel('x')
  plt.title('test')
  line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                     interval=50, blit=True)

  # To save the animation, use the command: line_ani.save('lines.mp4')

  import pdb; pdb.set_trace()
  # To save this second animation with some metadata, use the following command:
  # im_ani.save('im.mp4', metadata={'artist':'Guido'})

plt.show()

class LivePlotter(object):
  def __init__(self, results):
    assert isinstance(results, dict)
    self.results = results


def live_plot(self, data, x, y, hue, hue_order, log_scale=True):

  ax = sns.lineplot(data=data_frame, x=x, y=y,
                    hue=hue, hue_order=hue_order)
  ax.lines[-1].set_linestyle('-')
  ax.legend()
  if logscale:
    plt.yscale('log')
  plt.xlabel(x)
  plt.ylabel(y)
  plt.title(title)


def main():
  args = parser.parse_args()
  results = {}
  for name in plot_names:
    if ResultDict.is_loadable(name, args.load_dir):
      results[name] = ResultDict.load(name, args.load_dir)
    else:
      raise Exception(f"Unalbe to find result name: {name}")

  df_loss = pd.DataFrame()
  for name, result in results.items():
    df_loss = df_loss.append(
      result.data_frame(name, ['test_num', 'step_num']), sort=True)

  df_grad = pd.DataFrame()
  for name, result in results.items():
    df_grad = df_grad.append(
      result.data_frame(name, ['test_num', 'step_num', 'track_num']), sort=True)
  df_grad = df_grad[df_grad['grad'].abs() > 0.00001]
  df_grad = df_grad[df_grad['grad'].abs() < 0.001]
  df_grad['grad'] = df_grad['grad'].abs()
  df_grad['update'] = df_grad['update'].abs()

  sns.set(color_codes=True)
  sns.set_style('white')
  # import pdb; pdb.set_trace()

  grouped = df_loss.groupby(['optimizer', 'step_num'])
  df_loss = grouped.mean().reset_index()

  grouped = df_grad.groupby(['optimizer', 'step_num'])
  df_grad = grouped.mean().reset_index()

  time_init = df_loss['walltime'].min()
  time_delay = 1
  time_ratio = 0.05


  plt.ion()
  num_loop = 0
  fig = plt.figure(figsize=(20, 13))
  axes = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

  def animate(i):
    for ax in axes:
      ax.clear()

    time = watch.touch('cumulative')
    time_ref = time_init + (time - time_delay) * time_ratio
    df = df_loss[df_loss['walltime'] <= time_ref]
    #df = data_frame[data_frame['step_num'] <= i]
    for j, name in enumerate(plot_names):
      y = df[df['optimizer'] == name]['loss']
      x = df[df['optimizer'] == name]['step_num']
      if name == 'LSTM-base':
        name = 'Proposed'
      axes[0].plot(x, y, label=name, color=f'C{j}')
      axes[0].set_xlim([0, 200])
      axes[0].set_ylim([0.2, 3])
      axes[0].legend()
      axes[0].set_yscale('log')
      axes[0].set_xlabel('Step_num')
      axes[0].set_ylabel('Loss')
      axes[0].set_title('Loss w.r.t. num_step')

    for j, name in enumerate(plot_names):
      y = df[df['optimizer'] == name]['loss']
      x = df[df['optimizer'] == name]['walltime']
      if name == 'LSTM-base':
        name = 'Proposed'
      axes[1].plot(x, y, label=name, color=f'C{j}')
      axes[1].set_ylim([0.2, 3])
      axes[1].legend()
      axes[1].set_yscale('log')
      axes[1].set_xlabel('Walltime')
      axes[1].set_ylabel('Loss')
      axes[1].set_title('Loss w.r.t. walltime')

    df = df_grad[df_loss['walltime'] <= time_ref]

    for j, name in enumerate(plot_names):
      y = df[df['optimizer'] == name]['grad']
      x = df[df['optimizer'] == name]['step_num']
      if name == 'LSTM-base':
        name = 'Proposed'
      axes[2].plot(x, y, label=name, color=f'C{j}')
      axes[2].set_xlim([0, 200])
      # axes[2].set_ylim([0.2, 3])
      axes[2].legend()
      axes[2].set_xlabel('Step_num')
      axes[2].set_ylabel('Gradient')
      axes[2].set_title('Gradient w.r.t. step_num')

    for j, name in enumerate(plot_names):
      y = df[df['optimizer'] == name]['update']    
      x = df[df['optimizer'] == name]['step_num']
      if name == 'LSTM-base':
        name = 'Proposed'
        y *= 0.1
      axes[3].plot(x, y, label=name, color=f'C{j}')
      axes[3].set_xlim([0, 200])
      # axes[3].set_ylim([0.2, 3])
      axes[3].legend()
      axes[3].set_xlabel('Step_num')
      axes[3].set_ylabel('Update')
      axes[3].set_title('Update w.r.t. step_num')




  watch = StopWatch('Realtime')
  ani = animation.FuncAnimation(fig, animate, interval=1)
  plt.show()

  import pdb; pdb.set_trace()



if __name__ =="__main__":
  main()
  # test3()
