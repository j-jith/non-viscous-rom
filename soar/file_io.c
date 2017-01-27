#include "globals.h"
//#include <yaml.h>

void read_mat_file(MPI_Comm comm, const char filename[], Mat *A)
{
    PetscInt n_rows = 0;
    PetscInt n_cols = 0;

    PetscPrintf(comm, "Reading matrix from %s ...\n", filename);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &viewer);

    MatCreate(comm, A);
    MatSetFromOptions(*A);
    MatLoad(*A, viewer);

    PetscViewerDestroy(&viewer);

    MatGetSize(*A, &n_rows, &n_cols);
    PetscPrintf(comm, "Shape: (%d, %d)\n", n_rows, n_cols);
}

void read_vec_file(MPI_Comm comm, const char filename[], Vec *b)
{
    PetscInt n_rows = 0;

    PetscPrintf(comm, "Reading vector from %s ...\n", filename);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &viewer);

    VecCreate(comm, b);
    VecSetFromOptions(*b);
    VecLoad(*b, viewer);

    PetscViewerDestroy(&viewer);

    VecGetSize(*b, &n_rows);
    PetscPrintf(comm, "Shape: (%d, )\n", n_rows);
}

void write_mat_file(MPI_Comm comm, const char filename[], Mat *A)
{
    PetscInt n_rows, n_cols;
    PetscPrintf(comm, "Writing matrix to %s ...\n", filename);

    MatGetSize(*A, &n_rows, &n_cols);
    PetscPrintf(comm, "Shape: (%d, %d)\n", n_rows, n_cols);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);

    MatView(*A, viewer);

    PetscViewerDestroy(&viewer);
}

void write_vec_file(MPI_Comm comm, const char filename[], Vec *b)
{
    PetscInt n_rows = 0;
    PetscPrintf(comm, "Writing vector to %s ...\n", filename);

    VecGetSize(*b, &n_rows);
    PetscPrintf(comm, "Shape: (%d, )\n", n_rows);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);

    VecView(*b, viewer);

    PetscViewerDestroy(&viewer);
}


PetscInt count_rows(const char filename[])
{
    FILE *fp;
    int ch;
    PetscInt rows = 0;

    fp = fopen(filename, "r");

    if(fp == NULL) // if file not found
    {
        printf("File %s not found.\n", filename);
        return 0;
    }

    while(!feof(fp)) // while not EOF
    {
        ch = fgetc(fp);
        if (ch == '\n') // if newline, increment row count 
        {
            rows++;
        }
    }
    fclose(fp);
    
    printf("File %s has %d rows.\n", filename, rows);
    return rows;
}

Fitter* read_fitter(const char fit_file[], const char weight_file[], PetscInt *fit_len)
{
    FILE *fp;
    Fitter *fit_list;
    PetscInt n_fitter, n_weights, i, j;

    n_fitter = count_rows(fit_file); // no. of fitters
    n_weights = count_rows(weight_file); // no. of weights (= number of frequency points)

    *fit_len = n_fitter;

    fit_list = (Fitter*)malloc(sizeof(Fitter)*n_fitter);

    // read fitter coefficients
    fp = fopen(fit_file, "r");
    for(i = 0; i < n_fitter; i++)
    {
        fscanf(fp, "%le %le %le ", &(fit_list[i].c1[0]), &(fit_list[i].c1[1]), &(fit_list[i].c1[2]));
        fscanf(fp, "%le %le %le\n", &(fit_list[i].c2[0]), &(fit_list[i].c2[1]), &(fit_list[i].c2[2]));

        fit_list[i].weights = (PetscReal*)malloc(sizeof(PetscReal)*n_weights);

    }
    fclose(fp);

    // read fitter weights
    fp = fopen(weight_file, "r");
    for(i = 0; i < n_weights; i++)
    {
        for(j = 0; j < n_fitter-1; j++)
        {
            fscanf(fp, "%le ", &(fit_list[j].weights[i]));
        }
        fscanf(fp, "%le\n", &(fit_list[j].weights[i]));
    }
    fclose(fp);

    return fit_list;
}

/*
int read_params(const char filename[], Params params)
{
    FILE *fh = fopen(filename, "r");
    yaml_parser_t parser;
    yaml_event_t token;

    // Initialize parser
    if(!yaml_parser_initialize(&parser))
        fputs("Failed to initialize parser!\n", stderr);
    if(fh == NULL)
        fputs("Failed to open file!\n", stderr);

    // Set input file
    yaml_parser_set_input_file(&parser, fh);

    do
    {
        if (!yaml_parser_parse(&parser, &event))
        {
            printf("Parser error %d\n", parser.error);
            exit(EXIT_FAILURE);
        }

        if (event.type == YAML_SCALAR_EVENT)
        {
            switch
        }
        else if (event.type == YAML_STREAM_END_EVENT)
            yaml_event_delete(&event);

    } while(event.type != YAML_STREAM_END_EVENT);
    yaml_event_delete(&event);

    // Clean-up
    yaml_parser_delete(&parser);
    fcolse(fh);

    return 0;
}
*/
