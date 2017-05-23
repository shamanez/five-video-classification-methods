import cairo
import sys
import numpy as np
import random
import time

class ToyDataset():
    """Class that defines a spatiotemporal dataset generator."""
    def __init__(self, batch_size, width, height, nb_frames=16, px_per_frame=0.005):
        """Construct."""
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.ctx = cairo.Context(self.surface)
        
        self.ctx.scale(width, height) # Normalizing the canvas

        self.batch_size = batch_size
        self.nb_frames = nb_frames
        self.px_per_frame = px_per_frame  # amount to change per frame

        self.surface_width = width
        self.surface_height = height

    def gen_frame(self, x, y, w, h):
        self.gen_background()
        self.gen_rect(x, y, w, h)

    def gen_background(self):
        # Build background. Hard-coded right now.
        pat = cairo.LinearGradient(0.0, 0.0, 0.0, 1.0)
        pat.add_color_stop_rgba(1, 0.7, 0, 0, 1)
        pat.add_color_stop_rgba(0, 0.9, 0.7, 0.2, 1)
        self.ctx.rectangle(0, 0, 1, 1) # Rectangle(x0, y0, x1, y1)
        self.ctx.set_source(pat)
        self.ctx.fill()

    def gen_rect(self, x, y, w, h):
        self.ctx.set_source_rgb(0.3, 0.2, 0.5) # Solid color
        self.ctx.rectangle(x, y, w, h)
        self.ctx.set_line_width(0.03)
        self.ctx.set_source_rgb(0.5, 0.5, 1)
        self.ctx.fill()
        self.ctx.close_path()

    def write_frame(self, fname='tmp.png'):
        self.surface.write_to_png(fname) # Output to PNG

    def gen_data(self):
        """A generator that returns sequences of frames."""

        while True:
            # TODO: These should be initialized as numpy arrays.
            batch_X = []
            batch_y = []

            for _ in range(self.batch_size):
                # Get a random label.
                label = random.choice(["shrink", "grow", "same"])

                # Set starting point of square.
                x = random.random()
                y = random.random()

                # Width and height shouldn't be less than the amount they can shrink.
                w = max(random.random() / 2, self.nb_frames * self.px_per_frame)
                h = max(random.random() / 2, self.nb_frames * self.px_per_frame)

                # Reset X.
                sequence_X = []

                # Build our sequence.
                for i in range(self.nb_frames):
                    # Generate a frame at current state.
                    self.gen_frame(x, y, w, h)

                    # Change it?
                    if label == 'grow':
                        # Change size.
                        w += self.px_per_frame
                        h += self.px_per_frame

                        # Change location.
                        x -= self.px_per_frame / 2
                        y -= self.px_per_frame / 2

                    elif label == 'shrink':
                        # Change size.
                        w -= self.px_per_frame
                        h -= self.px_per_frame
                        
                        # Change location.
                        x += self.px_per_frame / 2
                        y += self.px_per_frame / 2

                    else:
                        # Do nothing.
                        pass

                    # Now add the new frame to our sequence as an array.
                    surface_arr = np.frombuffer(self.surface.get_data(), np.uint8)
                    surface_arr.shape = (self.surface_height, self.surface_width, 4)
                    surface_arr = surface_arr[:,:,:3]  # remove alpha channel

                    # Process the image for the NN.
                    surface_arr = (surface_arr / 255.).astype(np.float32)

                    # Add it to the array.
                    sequence_X.append(surface_arr)

                # Now that we have our sequence, get our y.
                # TEMP: This needs to be encoded properly.
                if label == 'grow':
                    sequence_y = [0, 1, 0]
                elif label == 'shrink':
                    sequence_y = [1, 0, 0]
                elif label == 'same':
                    sequence_y = [0, 0, 1]

                batch_X.append(sequence_X)
                batch_y.append(sequence_y)

            yield np.array(batch_X), np.array(batch_y)
    

if __name__ == '__main__':
    from PIL import Image
    toy = ToyDataset(32, 200, 100, 16)

    start = time.time()
    i = 0
    for x, y in toy.gen_data():
        print(x.shape, y.shape)

        # Checkout the first batch.
        for sequence in x:
            for i, image in enumerate(sequence):
                image *= 255
                image = Image.fromarray(image.astype('uint8'))
                image.save(str(i) + '.png')
            sys.exit()

        i += 1

        if i !=0 and i % 100 == 0:
            print(100 / ((time.time() - start) / 60))
            start = time.time()
            sys.exit()
