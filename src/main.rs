#![feature(stdarch_x86_avx512)]

use clap::Parser;
use libc::{c_void, madvise, memset, MADV_HUGEPAGE, MADV_POPULATE_WRITE, MADV_SEQUENTIAL};
use nix::sys::mman::{mmap, MapFlags, ProtFlags};
use rayon::prelude::*;
use std::fs::File;
use std::os::fd::IntoRawFd;
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
    #[arg(
        short = 'p',
        long,
        default_value = "false",
        value_name = "USE_MADV_POPULATE"
    )]
    pub use_madv_populate: Option<bool>,
    #[arg(short = 'm', long, value_name = "USE_MEMFD")]
    pub use_memfd: Option<bool>,
    #[arg(short = 'l', long, value_name = "USE_HUGE_PAGES")]
    pub use_huge_pages: Option<bool>,
    #[arg(short = 'c', long, value_name = "BENCHMARK_ONLY_MEMCPY")]
    pub benchmark_only_memcpy: Option<bool>,
    #[arg(short = 't', long, value_name = "USE_TRANSPARENT_HUGE_PAGES")]
    pub use_transparent_huge_pages: Option<bool>,
}

unsafe fn get_pointer_to_region_backing_fd(
    fd: i32,
    madv_populate: bool,
    use_memfd: bool,
    use_huge_pages: bool,
    no_threads: i32,
    use_transparent_huge_pages: bool,
) -> *mut c_void {
    // Mmap a region backed by a file.
    let mut addr = mmap(
        std::ptr::null_mut(),
        TO_WRITE,
        ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
        match fd {
            -1 => match use_huge_pages {
                true => MapFlags::MAP_ANONYMOUS | MapFlags::MAP_SHARED | MapFlags::MAP_HUGETLB,
                false => MapFlags::MAP_ANONYMOUS | MapFlags::MAP_SHARED,
            },
            _ => match use_memfd {
                true => match use_huge_pages {
                    true => MapFlags::MAP_SHARED | MapFlags::MAP_HUGETLB,
                    false => MapFlags::MAP_SHARED,
                },
                false => MapFlags::MAP_SHARED,
            },
        },
        fd,
        0,
    )
    .expect("Mmap failed");

    println!(
        "Address: {:?} is aligned for 2M {:?} or for 4K {:?}",
        addr as usize,
        (addr as usize) % (2 * MB) == 0,
        (addr as usize) % PAGE_SIZE == 0
    );

    if use_transparent_huge_pages {
        let ret = madvise(addr, TO_WRITE, MADV_HUGEPAGE);
        match ret {
            0 => {}
            _ => {
                println!("Madvise huge pages failed");
            }
        }
    }

    if madv_populate {
        let begin = time::Instant::now();
        // Start 8 threads that perform madvise on respective portions of memory.
        // This is done to ensure that the memory is actually allocated.
        // transform mmap_ptr to 64 bit int.
        let mmap_addr = addr as usize;
        let thread_chunk = TO_WRITE / no_threads as usize;
        let thread_vec: Vec<std::thread::JoinHandle<()>> = (0..no_threads)
            .map(|i| {
                let mmap_addr = mmap_addr + (i as usize * thread_chunk as usize);
                let mmap_size = thread_chunk as usize;
                std::thread::spawn(move || unsafe {
                    let local_ptr = mmap_addr as *mut c_void;
                    println!("madvise on {:?} with size {}", local_ptr, mmap_size);
                    let ret = madvise(local_ptr, mmap_size, MADV_POPULATE_WRITE);
                    match ret {
                        0 => {}
                        _ => {
                            println!("Madvise failed");
                        }
                    }
                })
            })
            .collect();

        for handle in thread_vec {
            handle.join().unwrap();
        }

        let after = begin.elapsed();
        println!(
            "Madvise POPULATE_WRITE took {:?} ms",
            after.as_millis() as usize
        );
    }
    addr
}

#[inline(always)]
pub fn manual_copy_from_slice(dst: &mut [u8], src: &[u8]) {
    if dst.len() != src.len() {
        panic!(
            "source and destination have different lengths: src has length {} and dst has length {}",
            src.len(),
            dst.len())
    };
    #[allow(clippy::manual_memcpy)]
    for i in 0..dst.len() {
        dst[i] = src[i];
    }
}

#[inline(always)]
pub fn fast_copy(src: *const u8, dst: *mut u8, len: usize) {
    for i in 0..len {
        unsafe {
            *dst.add(i) = *src.add(i);
        }
    }
}

unsafe fn measure_memory_copy() {
    unsafe {
        let addr1 = get_pointer_to_region_backing_fd(-1, true, true, false, 1, true);
        let addr2 = get_pointer_to_region_backing_fd(-1, true, true, false, 1, true);

        for i in 0..TO_WRITE {
            let addr1_ptr = addr1.offset(i as isize) as *mut u8;
            *addr1_ptr = 143;
        }

        let begin = time::Instant::now();

        // Copy page by page.
        for i in 0..(TO_WRITE / PAGE_SIZE) {
            let addr1_ptr = addr1.offset(i as isize * PAGE_SIZE as isize) as *mut u8;
            let addr2_ptr = addr2.offset(i as isize * PAGE_SIZE as isize) as *mut u8;

            //std::ptr::copy_nonoverlapping(addr1_ptr, addr2_ptr, PAGE_SIZE);

            // libc::memcpy(
            //     addr2_ptr as *mut c_void,
            //     addr1_ptr as *mut c_void,
            //     PAGE_SIZE,
            // );

            // for j in 0..PAGE_SIZE {
            //     *addr2_ptr.add(j) = *addr1_ptr.add(j);
            // }

            //fast_copy(addr1_ptr, addr2_ptr, PAGE_SIZE);
            manual_copy_from_slice(
                std::slice::from_raw_parts_mut(addr2_ptr, PAGE_SIZE),
                std::slice::from_raw_parts(addr1_ptr, PAGE_SIZE),
            );

            // const STRIDE: usize = 32;
            // for j in 0..PAGE_SIZE / STRIDE {

            //     let src = addr1_ptr.add(j * STRIDE);
            //     let dst = addr2_ptr.add(j * STRIDE);
            //     std::arch::asm! {
            //         "prefetchnta [{src} + 256]",
            //         "vmovntdqa ymm0, ymmword ptr [{src}]",
            //         "vmovntdq ymmword ptr [{dst}], ymm0",
            //         src = in(reg) src,
            //         dst = in(reg) dst,
            //     }
            // }
        }
        // std::arch::asm! {
        //     "sfence"
        // }

        let time_spent = begin.elapsed().as_millis() as usize;
        println!(
            "Memcpy took {:?} ms, bandwidth = {:?} MB/s",
            time_spent,
            (TO_WRITE / 1024) as f32 / time_spent as f32
        );
        for i in 0..TO_WRITE {
            let addr1_ptr = addr1.offset(i as isize) as *mut u8;
            let addr2_ptr = addr2.offset(i as isize) as *mut u8;
            if *addr1_ptr != *addr2_ptr {
                println!("Mismatch at index {}", i);
                break;
            }
        }
    }
}
fn main() {
    // Parse command line arguments.
    let args = PageFaultPerfArgs::parse();

    // Check if the flag for benchmarking memcpy is set.
    if args.benchmark_only_memcpy.unwrap_or(false) {
        unsafe { measure_memory_copy() };
        return;
    }

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
            let opts = memfd::MemfdOptions::default(); //.hugetlb(Some(memfd::HugetlbSize::Huge2MB));
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

    let use_huge_pages = match args.use_huge_pages {
        Some(use_huge_pages) => use_huge_pages,
        None => false,
    };

    let use_transparent_huge_pages = match args.use_transparent_huge_pages {
        Some(use_transparent_huge) => use_transparent_huge,
        None => false,
    };

    // Get pointer to the region backed by the file.
    let addr = unsafe {
        get_pointer_to_region_backing_fd(
            fd,
            use_madv_populate,
            use_memfd,
            use_huge_pages,
            no_threads,
            use_transparent_huge_pages,
        )
    };
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
