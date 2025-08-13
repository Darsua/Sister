typedef struct Pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

typedef struct Frame {
    Pixel *data;
    
    int width;
    int height;
    
    Pixel **pixel; // Pointer to each row of pixels for easier access (I felt so smart when I thought of this)
} Frame;

#define CHANNELS 3

Frame create_frame(int w, int h);

void render_frame(Frame* img, unsigned short** ref);

void save_frame(Frame* img);