#include <stdio.h>
#include <string.h>

void cli();
void gui();

int main(int argc, char *argv[]) {

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <mode>\n", argv[0]);
        fprintf(stderr, "Modes: CLI, GUI\n");
        return 1;
    }

    if (strcmp(argv[1], "GUI") == 0) {
        gui();
    } else if (strcmp(argv[1], "CLI") == 0) {
        cli();
    } else {
        fprintf(stderr, "Invalid mode: %s\n", argv[1]);
        fprintf(stderr, "Please choose either CLI or GUI.\n");
        return 1;
    }

    return 0;
}