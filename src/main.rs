use libc::{
    c_void, madvise, munmap, MADV_DONTNEED, MADV_HUGEPAGE, MADV_POPULATE_READ, MADV_POPULATE_WRITE,
    MADV_REMOVE, MADV_SEQUENTIAL,
};

use nix::sys::mman::{mmap, MapFlags, ProtFlags};

use clap::{Parser, Subcommand};
use memfd;
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::os::fd::IntoRawFd;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::time;

const MB: usize = 1024 * 1024;
const GB: usize = 1024 * MB;

const TO_WRITE: usize = 4 * GB;
const PAGE_SIZE: usize = 4096;

#[derive(Parser)]
#[command(
    version,
    about = "Page fault performance test.",
    long_about = "Page fault performance test."
)]
pub struct PageFaultPerfArgs {
    #[arg(short, long, value_name = "FILE_PATH")]
    pub file_path: Option<String>,
    #[arg(short, long, default_value = "1", value_name = "NO_THREADS")]
    pub no_threads: Option<i32>,
    #[arg(short, long, default_value = "false", value_name = "USE_MADV_POPULATE")]
    pub use_madv_populate: Option<bool>,
    #[arg(short, long, value_name = "USE_MEMFD")]
    pub use_memfd: Option<bool>,
}

unsafe fn get_pointer_to_region_backing_fd(
    fd: i32,
    madv_populate: bool,
    use_memfd: bool,
) -> *mut c_void {
    // Mmap a region backed by a file.
    let addr = mmap(
        std::ptr::null_mut(),
        TO_WRITE,
        ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
        match fd {
            -1 => MapFlags::MAP_ANONYMOUS | MapFlags::MAP_SHARED | MapFlags::MAP_HUGETLB,
            _ => match use_memfd {
                true => MapFlags::MAP_SHARED | MapFlags::MAP_ANONYMOUS | MapFlags::MAP_HUGETLB,
                false => MapFlags::MAP_SHARED,
            },
        },
        fd,
        0,
    )
    .expect("Mmap failed");

    if madv_populate {
        let begin = time::Instant::now();
        let ret = madvise(addr, TO_WRITE, MADV_POPULATE_WRITE);
        match ret {
            0 => {}
            _ => {
                println!("Madvise failed");
            }
        }
        let after = begin.elapsed();
        println!(
            "Madvise POPULATE_WRITE took {:?} ms",
            after.as_millis() as usize
        );
    }
    addr
}

fn main() {
    // Parse command line arguments.
    let args = PageFaultPerfArgs::parse();

    // If file path is none and memfd is none, then anonymous memory is needed and fd = -1.
    // A bad combination of parameters will result in fd = -2.
    println!("File path: {:?}", args.file_path);
    println!("Use memfd: {:?}", args.use_memfd);

    let fd = match (args.file_path, args.use_memfd) {
        (None, None) => {
            // Need to use anonymous memory.
            -1
        }
        (Some(file_path), None) => {
            // Need to use file-backed memory.
            let path = Path::new(&file_path);
            println!("Using file: {:?}", path);
            match File::options()
                .create(true)
                .read(true)
                .write(true)
                .open(&file_path)
            {
                Ok(file) => file.into_raw_fd(),
                Err(e) => {
                    println!("Error creating file: {:?}", e);
                    -1
                }
            }
        }
        (None, Some(true)) => {
            // Create an memfd file.
            let opts = memfd::MemfdOptions::default().hugetlb(Some(memfd::HugetlbSize::Huge2MB));
            match opts.create("") {
                Ok(memfd) => memfd.into_raw_fd(),
                Err(e) => {
                    println!("Error creating memfd: {:?}", e);
                    -1
                }
            }
        }
        (_, _) => -2,
    };

    if fd == -2 {
        println!("Invalid arguments");
        return;
    }

    if fd > 0 {
        // Need to make the file the right size for memmapping.
        unsafe {
            libc::ftruncate(fd, TO_WRITE as i64);
        }
    }

    let no_threads = match args.no_threads {
        Some(no_threads) => no_threads,
        None => 1,
    };

    let use_madv_populate = match args.use_madv_populate {
        Some(use_madv_populate) => use_madv_populate,
        None => false,
    };

    let use_memfd = match args.use_memfd {
        Some(use_memfd) => use_memfd,
        None => false,
    };

    // Get pointer to the region backed by the file.
    let addr = unsafe { get_pointer_to_region_backing_fd(fd, use_madv_populate, use_memfd) };
    let ptr_u8 = (addr as usize) as *mut u8;

    // Collect in an array the addresses of each page in the region.
    let mut addresses_vec: Vec<u64> = Vec::new();
    for i in 0..(TO_WRITE / PAGE_SIZE) {
        let page_addr = unsafe { ptr_u8.offset((i * PAGE_SIZE) as isize) as u64 };
        addresses_vec.push(page_addr);
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(no_threads as usize)
        .build_global()
        .unwrap();

    let begin = time::Instant::now();
    addresses_vec.par_iter().for_each(|address| {
        // Touch each page once.
        let address_ptr = *address as *mut u8;
        unsafe {
            *address_ptr.add(13) += 1;
        }
    });
    let after = begin.elapsed();
    println!(
        "Touching all pages took {:?} ms",
        after.as_millis() as usize
    );
}