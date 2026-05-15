"""
Demonstrate the use of multiprocessing with PyMuPDF.

Depending on the  number of CPUs, the document is divided in page ranges.
Each range is then worked on by one process.
The type of work would typically be text extraction or page rendering. Each
process must know where to put its results, because this processing pattern
does not include inter-process communication or data sharing.

Compared to sequential processing, speed improvements in range of 100% (ie.
twice as fast) or better can be expected.
"""
from __future__ import print_function, division
import sys
import os
import time
from multiprocessing import Pool, cpu_count
import pymupdf
import io


def render_page(args):
    """Render a page range of a PDF document.

    Notes:
        The PyMuPDF document cannot be part of the argument, because that
        cannot be pickled. So we are being passed in just its filename.
        This is no performance issue, because we are a separate process and
        need to open the document anyway.
        Any page-specific function can be processed here - rendering is just
        an example - text extraction might be another.
        The work must however be self-contained: no inter-process communication or synchronization is possible with this design.
        Care must also be taken with which parameters are contained in the
        argument, because it will be passed in via pickling by the Pool class.
        So any large objects will increase the overall duration.
    Args:
        args: a list containing required parameters.
        args[0] is the segment number we have to process
        args[1] is the number of CPUs
        args[2] is the document filename
        args[3] is the matrix for rendering
    """
    # recreate the arguments
    idx = args[0]  # this is the segment number we have to process
    cpu = args[1]  # number of CPUs
    filename = args[2]  # document filename
    mat = args[3]  # the matrix for rendering
    doc = pymupdf.open(filename)  # open the document
    num_pages = doc.page_count  # get number of pages

    # pages per segment: make sure that cpu * seg_size >= num_pages!
    seg_size = int(num_pages / cpu + 1)
    seg_from = idx * seg_size  # our first page number
    seg_to = min(seg_from + seg_size, num_pages)  # last page number

    pages = []
    for i in range(seg_from, seg_to):  # work through our page segment
        page = doc[i]
        pix = page.get_pixmap(alpha=False, matrix=mat)
        out = pix.tobytes("png")
        pages.append(out)
        # help release memory promptly
        pix = None
        page = None

    doc.close()
    return pages 


def parse_pdf_pages(filename, n_workers=None, dpi=300) -> list:
    t0 = time.time()  # start the timer
    mat = pymupdf.Matrix(dpi/72, dpi/72)
    cpu = n_workers if n_workers is not None else cpu_count()

    # make vectors of arguments for the processes
    args = [(i, cpu, filename, mat) for i in range(cpu)]
    print(f"Starting {cpu} processes for '{filename}'.")

    pool = Pool()  # make pool of 'cpu_count()' processes
    pages = pool.map(render_page, args, 1)

    t1 = time.time()  # stop the timer
    print(f"Total time {round(t1 - t0, 2):g} seconds")
    return pages

if __name__ == "__main__":
    filename = 'data/officeqa/treasury_bulletin_pdfs/treasury_bulletin_1939_01.pdf'
    pages = [p for x in parse_pdf_pages(filename, n_workers=4, dpi=300) for p in x]
    print(pages[80][:100])
    print(f"Total pages: {len(pages)}")