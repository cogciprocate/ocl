    //############################### NEW 2.1 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clSetDefaultDeviceCommandQueue(cl_context           /* context */,
                                   cl_device_id         /* device */,
                                   cl_command_queue     /* command_queue */) CL_API_SUFFIX__VERSION_2_1;

    //############################### NEW 2.1 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clGetDeviceAndHostTimer(cl_device_id    /* device */,
                            cl_ulong*       /* device_timestamp */,
                            cl_ulong*       /* host_timestamp */) CL_API_SUFFIX__VERSION_2_1;

    //############################### NEW 2.1 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clGetHostTimer(cl_device_id /* device */,
                   cl_ulong *   /* host_timestamp */)  CL_API_SUFFIX__VERSION_2_1;

    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_command_queue CL_API_CALL
    clCreateCommandQueueWithProperties(cl_context               /* context */,
                                       cl_device_id             /* device */,
                                       const cl_queue_properties *    /* properties */,
                                       cl_int *                 /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;


    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_mem CL_API_CALL
    clCreatePipe(cl_context                 /* context */,
                 cl_mem_flags               /* flags */,
                 cl_uint                    /* pipe_packet_size */,
                 cl_uint                    /* pipe_max_packets */,
                 const cl_pipe_properties * /* properties */,
                 cl_int *                   /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;


    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clGetPipeInfo(cl_mem           /* pipe */,
                  cl_pipe_info     /* param_name */,
                  size_t           /* param_value_size */,
                  void *           /* param_value */,
                  size_t *         /* param_value_size_ret */) CL_API_SUFFIX__VERSION_2_0;

    // SVM Allocation APIs 
    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY void * CL_API_CALL
    clSVMAlloc(cl_context       /* context */,
               cl_svm_mem_flags /* flags */,
               size_t           /* size */,
               cl_uint          /* alignment */) CL_API_SUFFIX__VERSION_2_0;

    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY void CL_API_CALL
    clSVMFree(cl_context        /* context */,
              void *            /* svm_pointer */) CL_API_SUFFIX__VERSION_2_0;


    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_sampler CL_API_CALL
    clCreateSamplerWithProperties(cl_context                     /* context */,
                                  const cl_sampler_properties *  /* normalized_coords */,
                                  cl_int *                       /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;

    //############################### NEW 2.1 #################################
    extern CL_API_ENTRY cl_program CL_API_CALL
    clCreateProgramWithIL(cl_context    /* context */,
                         const void*    /* il */,
                         size_t         /* length */,
                         cl_int*        /* errcode_ret */) CL_API_SUFFIX__VERSION_2_1;

    //############################### NEW 2.1 #################################
    extern CL_API_ENTRY cl_kernel CL_API_CALL
    clCloneKernel(cl_kernel     /* source_kernel */,
                  cl_int*       /* errcode_ret */) CL_API_SUFFIX__VERSION_2_1;

    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clSetKernelArgSVMPointer(cl_kernel    /* kernel */,
                             cl_uint      /* arg_index */,
                             const void * /* arg_value */) CL_API_SUFFIX__VERSION_2_0;

    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clSetKernelExecInfo(cl_kernel            /* kernel */,
                        cl_kernel_exec_info  /* param_name */,
                        size_t               /* param_value_size */,
                        const void *         /* param_value */) CL_API_SUFFIX__VERSION_2_0;


    //############################### NEW 2.1 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clGetKernelSubGroupInfo(cl_kernel                   /* kernel */,
                            cl_device_id                /* device */,
                            cl_kernel_sub_group_info    /* param_name */,
                            size_t                      /* input_value_size */,
                            const void*                 /*input_value */,
                            size_t                      /* param_value_size */,
                            void*                       /* param_value */,
                            size_t*                     /* param_value_size_ret */ ) CL_API_SUFFIX__VERSION_2_1;


    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clEnqueueSVMFree(cl_command_queue  /* command_queue */,
                     cl_uint           /* num_svm_pointers */,
                     void *[]          /* svm_pointers[] */,
                     void (CL_CALLBACK * /*pfn_free_func*/)(cl_command_queue /* queue */,
                                                            cl_uint          /* num_svm_pointers */,
                                                            void *[]         /* svm_pointers[] */,
                                                            void *           /* user_data */),
                     void *            /* user_data */,
                     cl_uint           /* num_events_in_wait_list */,
                     const cl_event *  /* event_wait_list */,
                     cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;

    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clEnqueueSVMMemcpy(cl_command_queue  /* command_queue */,
                       cl_bool           /* blocking_copy */,
                       void *            /* dst_ptr */,
                       const void *      /* src_ptr */,
                       size_t            /* size */,
                       cl_uint           /* num_events_in_wait_list */,
                       const cl_event *  /* event_wait_list */,
                       cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;

    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clEnqueueSVMMemFill(cl_command_queue  /* command_queue */,
                        void *            /* svm_ptr */,
                        const void *      /* pattern */,
                        size_t            /* pattern_size */,
                        size_t            /* size */,
                        cl_uint           /* num_events_in_wait_list */,
                        const cl_event *  /* event_wait_list */,
                        cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;

    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clEnqueueSVMMap(cl_command_queue  /* command_queue */,
                    cl_bool           /* blocking_map */,
                    cl_map_flags      /* flags */,
                    void *            /* svm_ptr */,
                    size_t            /* size */,
                    cl_uint           /* num_events_in_wait_list */,
                    const cl_event *  /* event_wait_list */,
                    cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;

    //############################### NEW 2.0 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clEnqueueSVMUnmap(cl_command_queue  /* command_queue */,
                      void *            /* svm_ptr */,
                      cl_uint           /* num_events_in_wait_list */,
                      const cl_event *  /* event_wait_list */,
                      cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;

    //############################### NEW 2.1 #################################
    extern CL_API_ENTRY cl_int CL_API_CALL
    clEnqueueSVMMigrateMem(cl_command_queue         /* command_queue */,
                           cl_uint                  /* num_svm_pointers */,
                           const void **            /* svm_pointers */,
                           const size_t *           /* sizes */,
                           cl_mem_migration_flags   /* flags */,
                           cl_uint                  /* num_events_in_wait_list */,
                           const cl_event *         /* event_wait_list */,
                           cl_event *               /* event */) CL_API_SUFFIX__VERSION_2_1;

