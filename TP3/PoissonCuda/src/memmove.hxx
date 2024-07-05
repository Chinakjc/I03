double * allocate(int n);
void deallocate(double *&d);

void copyDeviceToHost
        (double *h, double *d, int n);
        
void copyHostToDevice
        (double *h, double *d, int n);

void copyDeviceToDevice
        (double *d_out, double *d_in, int n);
