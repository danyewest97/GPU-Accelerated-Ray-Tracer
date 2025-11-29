// Sorry if I use snake case for Java, it gets confusing switching between HIP and Java and naming stuff -- I might go back and rectify names to 
// follow convention later but it isn't a big issue in my eyes so we'll see

// Loading the necessary libraries for drawing the output image
import java.util.*;
import java.awt.*;
import java.awt.Color;                      // Specifying which Color class to use (unsure why this was an issue, there should only be one, but oh 
                                            // well -- maybe util has its own Color class that I didn't know about)
import java.awt.image.*;
import javax.swing.*;

public class Main {
    public static int width = 40;
    public static int height = 40;
    public static double[] output = null;
    public native double[] test(int width, int height);                // Declaring a native function name -- native = from a dll/other coding 
                                                                       // language

    // Runs when the class is loaded (aka immediately after compilation)
    static {
        System.loadLibrary("native");                          // Loading the JNI library that allows us to mesh Java and C++
                                                                // JNI = Java Native Interface
    }

    // Had to rename the Graphics object g to gr so that it didn't mess with the green component name lol
    public static void draw(Graphics gr) {
        if (output != null) {
            BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < output.length; i += 3) {
                // Getting the generated color values from the GPU
                double r = output[i];
                double g = output[i + 1];
                double b = output[i + 2];

                // Normalising the color values to fall between 0 and 1
                r = Math.max(0, Math.min(1, r));
                g = Math.max(0, Math.min(1, g));
                b = Math.max(0, Math.min(1, b));

                // Converting the double color values to integer color values (may switch BufferedImage color mode to circumvent this later)
                // Adding 0.5 to round the color values to the nearest integer
                int r_int =  255 * (int) (r + 0.5);
                int g_int =  255 * (int) (g + 0.5);
                int b_int =  255 * (int) (b + 0.5);

                // Calculating the image x- and y-values from the index of the pixel we are on (i / 3 because there are 3 color values for each pixel)
                int pixel_idx = i / 3;
                int x = pixel_idx % width;
                int y = pixel_idx / width;

                Color pixel_color = new Color(r_int, g_int, b_int);                     // Creating a new color with our new integer color values
                int pixel_color_rgb_int_value = pixel_color.getRGB();                   // Taking the RGB of this new color, as an integer, which is  
                                                                                        // needed to pass it to setRGB()
                img.setRGB(x, y, pixel_color_rgb_int_value);                            // Setting the color of the final pixel in our image
            }
            
            // Finally, drawing the generated BufferedImage to our JPanel
            // Coordinates on the JPanel where the image is drawn -- set to (0, 0) because we want the image to fill the whole JPanel with no offset
            int img_x = 0;
            int img_y = 0;

            // A placeholder variable for the ImageObserver argument in drawImage, not really sure what ImageObservers are or how they work but we 
            // don't need to specify an observer in this case, it's fine to leave null
            ImageObserver observer = null;

            // Drawing the image! :D
            gr.drawImage(img, img_x, img_y, observer);
        }
    }

    public static JFrame createFrame(int width, int height) {
        JFrame frame = new JFrame("not your average window");                  // Creating a new JFrame (aka a new window) with the given name
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);                  // Making the window terminate the program when closed
        frame.setSize(width, height);                                           // Setting the size of the window to the size of the image being drawn
        frame.setVisible(true);                                              // Making the window visible
        return frame;
    }

    public static JPanel createPanel(JFrame parent) {
        // Creating a new JPanel to draw the image to that becomes a child of the parent JFrame
        // As far as I know, the convention is to use JPanels to draw stuff (you can draw to the JFrame directly but it isn't usually done) and have a 
        // JFrame just as the parent window
        // This especially helps for organization when you have multiple overlapping elements/JPanels, but we just have one so it isn't totally 
        // necessary, however I think that JPanels are more convenient, faster, and easier to clear than JFrames so that's why I'm using one here
        // And, as previously stated, it's the convention for stuff like this and it's how I've been doing it since the start
        JPanel panel = new JPanel() {
                                                           // Overriding the paint method of the JPanel class -- this will get called each 
            @Override                                      // time the panel is "painted," or updated on screen
            public void paintComponent(Graphics gr) {
                draw(gr);                                  // Redirecting to the draw() function for easier organization in the code
            }
        };

        panel.setPreferredSize(new Dimension(width, height)); // We need to use preferred size instead of size here because pack() only pays attention 
                                                              // to preferred sizes, so the parent frame will not readjust its size unless we set the 
                                                              // preferred size
        parent.add(panel);                                 // Adding the newly-made panel to the parent JFrame that it will be drawn on
        parent.pack();                                     // Resetting the automatic sizing of the JFrame window to fit the new JPanel
        return panel;
    }

    public static void main(String[] args) {
        System.out.println("Running!");
        JFrame frame = createFrame(width, height);
        JPanel panel = createPanel(frame);

        // Note: DO NOT PUT THIS IN A LOOP OR TIMER WITHOUT MAKING SOME SORT OF TERMINATION SAFETY!! THE GPU CAN CRASH WHEN TERMINATING PREMATURELY!!
        //for (int i = 0; i < 100; i++) {
            output = new Main().test(width, height);
        // for (double color_value : output) {
        //     System.out.println(color_value);
        // }
            panel.repaint();
        //}

        System.out.println("\nProgram finished!");
    }
}
