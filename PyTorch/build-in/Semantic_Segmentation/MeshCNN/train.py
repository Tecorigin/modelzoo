import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # 每个step都打印loss
            loss = model.loss
            t = (time.time() - iter_start_time) / opt.batch_size
            print('Epoch: %d, Iter: %d, Step: %d, Loss: %.6f, Time: %.6f, Data Time: %.6f' % 
                  (epoch, i, total_steps, loss, t, t_data))
            # 如果不想使用writer打印，可以注释掉下面两行
            # writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
            # writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            iter_data_time = time.time()
            
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch)
            writer.plot_acc(acc, epoch)

    writer.close()