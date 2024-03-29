# code referenced from https://github.com/vincentherrmann/pytorch-wavenet/blob/master/model_logging.py
import threading
from io import BytesIO
import time

import tensorflow as tf
import numpy as np
from PIL import Image
import torch


class Logger:
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 info_interval=1000,
                 trainer=None,
                 generate_function=None):
        self.trainer = trainer
        self.log_interval = log_interval
        self.val_interval = validation_interval
        self.gen_interval = generate_interval
        self.info_interval = info_interval
        self.log_time = time.time()
        self.accumulated_loss = 0
        self.accumulated_ce_loss = 0
        self.accumulated_bitperchar = 0
        self.generate_function = generate_function
        if self.generate_function is not None:
            self.generate_thread = threading.Thread(target=self.generate_function)
            self.generate_function.daemon = True

    def log(self, current_step, current_losses, current_grad_norm):
        self.accumulated_loss += float(current_losses['loss'].detach())
        self.accumulated_ce_loss += float(current_losses['ce_loss'].detach())
        if 'bitperchar' in current_losses:
            self.accumulated_bitperchar += float(current_losses['bitperchar'].detach())
        if current_step % self.log_interval == 0:
            self.log_loss(current_step)
            self.log_time = time.time()
            self.accumulated_loss = 0
            self.accumulated_ce_loss = 0
            self.accumulated_bitperchar = 0
        if self.val_interval is not None and self.val_interval > 0 and current_step % self.val_interval == 0:
            self.validate(current_step)
        if self.gen_interval is not None and self.gen_interval > 0 and current_step % self.gen_interval == 0:
            self.generate(current_step)
        if self.info_interval is not None and self.info_interval > 0 and current_step % self.info_interval == 0:
            self.info(current_step)

    def log_loss(self, current_step):
        avg_loss = self.accumulated_loss / self.log_interval
        avg_ce_loss = self.accumulated_ce_loss / self.log_interval
        avg_bitperchar = self.accumulated_bitperchar / self.log_interval
        print(f"{time.time()-self.log_time:7.3f} loss, ce_loss, bitperchar at step {current_step:8d}: "
              f"{avg_loss:11.6f}, {avg_ce_loss:11.6f}, {avg_bitperchar:10.6f}", flush=True)

    def validate(self, current_step):
        validation = self.trainer.validate()
        if validation is None:
            return
        losses, accuracies, true_outputs, logits, rocs = validation
        print(f"validation losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}", flush=True)
        print(f"validation accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}", flush=True)
        # print(f"validation true values: {', '.join(['{:6.4f}'.format(val) for val in true_outputs])}", flush=True)
        print(f"validation average logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}", flush=True)
        print(f"validation AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}", flush=True)

    def generate(self, current_step):
        if self.generate_function is None:
            return

        if self.generate_thread.is_alive():
            print("Last generate is still running, skipping this one")
        else:
            self.generate_thread = threading.Thread(target=self.generate_function, args=[current_step])
            self.generate_thread.daemon = True
            self.generate_thread.start()

    def info(self, current_step):
        pass
        # print(
        #     'GPU Mem Allocated:',
        #     round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1),
        #     'GB, ',
        #     'Cached:',
        #     round(torch.cuda.memory_cached(0) / 1024 ** 3, 1),
        #     'GB'
        # )


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class TensorboardLogger(Logger):
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 info_interval=1000,
                 trainer=None,
                 generate_function=None,
                 log_dir='logs',
                 log_param_histograms=False,
                 log_image_summaries=True,
                 print_output=False,
                 ):
        super().__init__(
            log_interval, validation_interval, generate_interval, info_interval, trainer, generate_function)
        self.writer = tf.summary.FileWriter(log_dir)
        self.log_param_histograms = log_param_histograms
        self.log_image_summaries = log_image_summaries
        self.print_output = print_output

    def log(self, current_step, current_losses, current_grad_norm):
        super(TensorboardLogger, self).log(current_step, current_losses, current_grad_norm)
        self.scalar_summary('grad norm', current_grad_norm, current_step)
        self.scalar_summary('loss', current_losses['loss'].detach(), current_step)
        self.scalar_summary('ce_loss', current_losses['ce_loss'].detach(), current_step)
        if 'accuracy' in current_losses:
            self.scalar_summary('accuracy', current_losses['accuracy'].detach(), current_step)
        if 'bitperchar' in current_losses:
            self.scalar_summary('bitperchar', current_losses['bitperchar'].detach(), current_step)

    def log_loss(self, current_step):
        if self.print_output:
            Logger.log_loss(self, current_step)
        # loss
        avg_loss = self.accumulated_loss / self.log_interval
        avg_ce_loss = self.accumulated_ce_loss / self.log_interval
        avg_bitperchar = self.accumulated_bitperchar / self.log_interval
        self.scalar_summary('avg loss', avg_loss, current_step)
        self.scalar_summary('avg ce loss', avg_ce_loss, current_step)
        self.scalar_summary('avg bitperchar', avg_bitperchar, current_step)

        if self.log_param_histograms:
            for tag, value, in self.trainer.model.named_parameters():
                tag = tag.replace('.', '/')
                self.histo_summary(tag, value.data, current_step)
                if value.grad is not None:
                    self.histo_summary(tag + '/grad', value.grad.data, current_step)

        if self.log_image_summaries:
            for tag, summary in self.trainer.model.image_summaries.items():
                self.image_summary(tag, summary['img'], current_step, max_outputs=summary.get('max_outputs', 3))

    def validate(self, current_step):
        validation = self.trainer.validate()
        if validation is None:
            return
        losses, accuracies, true_outputs, logits, rocs = validation
        for i, loss, acc in enumerate(zip(losses, accuracies)):
            self.scalar_summary(f'validation loss {i}', loss, current_step)
            self.scalar_summary(f'validation accuracy {i}', acc, current_step)
        if self.print_output:
            print(f"validation losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}", flush=True)
            print(f"validation accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}",
                  flush=True)
            # print(f"validation true values: {', '.join(['{:6.4f}'.format(val) for val in true_outputs])}", flush=True)
            print(f"validation average logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}", flush=True)
            print(f"validation AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}", flush=True)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if isinstance(value, torch.Tensor):
            value = value.item()  # value must have 1 element only
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step, max_outputs=3):
        """Log a tensor image.
        :param tag: string summary name
        :param images: (N, H, W, C) or (N, H, W)
        :param step: current step
        :param max_outputs: max N images to save
        """

        img_summaries = []
        for i in range(min(images.size(0), max_outputs)):
            img = images[i].cpu().numpy()

            # Write the image to a string
            s = BytesIO()
            Image.fromarray(img, 'RGB').save(s, format="png")
            # scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=200):
        """Log a histogram of the tensor of values."""
        values = values.cpu().numpy()

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def tensor_summary(self, tag, tensor, step):
        tf_tensor = tf.Variable(tensor).to_proto()
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, tensor=tf_tensor)])
        # summary = tf.summary.tensor_summary(name=tag, tensor=tensor)
        self.writer.add_summary(summary, step)
