use std::thread::{self, JoinHandle};
use futures::{executor, SinkExt, StreamExt, Future};
use futures::channel::mpsc::{self, Sender};
use futures_cpupool::{CpuPool, Builder as CpuPoolBuilder};
use tokio_core::reactor::Core;
#[cfg(feature = "profile")]
use cpuprofiler::PROFILER;
use cmn::{CmnResult};

/// A pool of worker threads and a tokio reactor core used to complete futures.
pub struct WorkPoolOld {
    cpu_pool: CpuPool,
    reactor_tx: Option<Sender<Box<Future<Item = (), Error = ()> + Send>>>,
    _reactor_thread: Option<JoinHandle<()>>,
}

impl WorkPoolOld {
    /// Returns a new `WorkPoolOld`.
    pub fn new(buffer_size: usize) -> WorkPoolOld {
        let (reactor_tx, reactor_rx) = mpsc::channel(0);
        let reactor_thread_name = "bismit_work_pool_reactor_core".to_owned();

        let reactor_thread: JoinHandle<_> = thread::Builder::new()
                .name(reactor_thread_name).spawn(move || {
            let mut core = Core::new().unwrap();
            let work = reactor_rx.buffer_unordered(buffer_size).for_each(|_| Ok(()));
            core.run(work).unwrap();
        }).unwrap();

        let cpu_pool = CpuPoolBuilder::new().name_prefix("bismit_work_pool_worker_").create();

        WorkPoolOld {
            cpu_pool,
            reactor_tx: Some(reactor_tx),
            _reactor_thread: Some(reactor_thread),
        }
    }

    /// Submits a future which need only be polled to completion and that
    /// contains no intensive CPU work (including memcpy).
    pub fn complete<F>(&mut self, future: F) -> CmnResult<()>
            where F: Future<Item=(), Error=()> + Send + 'static {
        let tx = self.reactor_tx.take().unwrap();
        self.reactor_tx.get_or_insert(executor::block_on(tx.send(Box::new(future)))?);
        Ok(())
    }

    /// Submit a future which contains non-trivial CPU work (including memcpy).
    pub fn complete_work<F>(&mut self, work: F) -> CmnResult<()>
            where F: Future<Item=(), Error=()> + Send + 'static {
        let future = self.cpu_pool.spawn(work);
        self.complete(future)
    }

    // /// Returns a remote to this `WorkPoolOld` usable to submit work.
    // pub fn remote(&self) -> WorkPoolOldRemote {
    //     WorkPoolOldRemote {
    //         cpu_pool: self.cpu_pool.clone(),
    //         reactor_tx: self.reactor_tx.clone(),
    //     }
    // }
}

impl Drop for WorkPoolOld {
    fn drop(&mut self) {
        self.reactor_tx.take().unwrap().close().unwrap();
        self._reactor_thread.take().unwrap().join().expect("error joining reactor thread");
    }
}


// /// A remote control for `WorkPoolOld` used to submit futures needing completion.
// #[derive(Clone)]
// pub struct WorkPoolOldRemote {
//     cpu_pool: CpuPool,
//     reactor_tx: Option<Sender<Box<Future<Item=(), Error=()> + Send>>>,
// }

// impl WorkPoolOldRemote {
//     /// Submits a future which need only be polled to completion and that
//     /// contains no intensive CPU work (including memcpy).
//     pub fn complete<F>(&mut self, future: F) -> CmnResult<()>
//             where F: Future<Item=(), Error=()> + Send + 'static {
//         let tx = self.reactor_tx.take().unwrap();
//         self.reactor_tx.get_or_insert(executor::block_on(tx.send(Box::new(future)))?);
//         Ok(())
//     }

//     /// Submit a future which contains non-trivial CPU work (including memcpy).
//     pub fn complete_work<F>(&mut self, work: F) -> CmnResult<()>
//             where F: Future<Item=(), Error=()> + Send + 'static {
//         let future = self.cpu_pool.spawn(work);
//         self.complete(future)
//     }
// }