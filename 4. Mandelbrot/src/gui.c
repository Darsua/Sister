#include <raylib.h>
#include <stddef.h>
#define RAYGUI_IMPLEMENTATION
#include "../include/raygui.h"
#include "../include/gpu.h"

Texture2D
compute_next(Frame *fr, float cr, float ci, float s)
{
    render_frame(fr, cuda_next(fr, cr, ci, s));
    save_frame(fr);
    return LoadTexture("mandelbrot.bmp");
}

int
gui()
{
    InitWindow(2000, 2000, "Mandelbruh");
    GuiSetStyle(DEFAULT, TEXT_SIZE, 24);
    SetTargetFPS(60);
    
    // GUI state
    float offsetX = 0.0;
    float offsetY = 0.0;
    float zoom = 1.0;
    bool dragging = false;
    Vector2 lastMouse = {0};
    int zooming = 0;
    const int ZOOM_PAUSE_FRAMES = 30;
    
    // Mandelbrot state
    Frame fr = create_frame(2000, 2000);
    double center_real = 0.0;
    double center_imag = 0.0;
    double scale = 1.0;
    Texture2D texture = compute_next(&fr, offsetX, offsetY, zoom);

    while (!WindowShouldClose())
    {
        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            zooming = 1;
        
            Vector2 mouse = GetMousePosition();
            float prevZoom = zoom;
            zoom *= pow(1.1, wheel);
            if (zoom < 0.1) zoom = 0.1;
        
            // Adjust offset so zoom centers on mouse
            offsetX = (offsetX - mouse.x) * (zoom / prevZoom) + mouse.x;
            offsetY = (offsetY - mouse.y) * (zoom / prevZoom) + mouse.y;
        }
        
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            dragging = true;
            lastMouse = GetMousePosition();
        }
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            dragging = false;
            
            center_real -= offsetX * 4.0 / (scale * fr.width);
            center_imag -= offsetY * 4.0 / (scale * fr.height);
            
            offsetX = 0.0;
            offsetY = 0.0;
            
            texture = compute_next(&fr, center_real, center_imag, scale);
        }
        if (dragging) {
            Vector2 mouse = GetMousePosition();
            offsetX += mouse.x - lastMouse.x;
            offsetY += mouse.y - lastMouse.y;
            lastMouse = mouse;
        }
        
        BeginDrawing();
            ClearBackground(BLACK);
            
            Rectangle src = {0, 0, texture.width, texture.height};
            Rectangle dest = {
                offsetX,
                offsetY,
                texture.width * zoom,
                texture.height * zoom
            };
            DrawTexturePro(texture, src, dest, (Vector2){0, 0}, 0.0, WHITE);
        EndDrawing();
        
        if (zooming > 0) {
            zooming++;
            if (zooming > ZOOM_PAUSE_FRAMES) {
                // Calculate new Mandelbrot parameters
                Vector2 mouse = GetMousePosition();
                
                // Complex coordinate under mouse before zoom
                double mouse_real_before = center_real + (mouse.x - fr.width / 2.0) * 4.0 / (scale * fr.width);
                double mouse_imag_before = center_imag + (mouse.y - fr.height / 2.0) * 4.0 / (scale * fr.height);
                
                // Update scale
                scale *= zoom;
                if (scale < 0.1) scale = 0.1;
        
                // Complex coordinate under mouse after zoom
                double mouse_real_after = center_real + (mouse.x - fr.width / 2.0) * 4.0 / (scale * fr.width);
                double mouse_imag_after = center_imag + (mouse.y - fr.height / 2.0) * 4.0 / (scale * fr.height);
        
                // Adjust center so mouse stays fixed
                center_real += mouse_real_before - mouse_real_after;
                center_imag += mouse_imag_before - mouse_imag_after;
        
                // Reset GUI zoom/offset
                zoom = 1.0;
                offsetX = 0.0;
                offsetY = 0.0;
        
                // Re-render
                texture = compute_next(&fr, center_real, center_imag, scale);
                zooming = 0;
            }
        }
    }

    CloseWindow();
    return 0;
}
