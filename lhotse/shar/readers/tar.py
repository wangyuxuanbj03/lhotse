import io
import tarfile
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

from lhotse import Features, Recording
from lhotse.array import Array, TemporalArray
from lhotse.serialization import decode_json_line, deserialize_item, open_best
from lhotse.shar.utils import fill_shar_placeholder
from lhotse.utils import Pathlike

Manifest = Union[Recording, Array, TemporalArray, Features]


class TarIterator:
    """
    TarIterator is a convenience class for reading arrays/audio stored in Lhotse Shar tar files.
    It is specific to Lhotse Shar format and expects the tar file to have the following structure:

    * each file is stored in a separate tar member
    * the file name is the key of the array
    * every array has two corresponding files:
        * the metadata: the file extension is ``.json`` and the file contains
          a Lhotse manifest (Recording, Array, TemporalArray, Features)
          for the data item.
        * the data: the file extension is the format of the array,
          and the file contents are the serialized array (possibly compressed).
        * the data file can be empty in case some cut did not contain that field.
          In that case, the data file has extension ``.nodata`` and the manifest file
          has extension ``.nometa``.
        * these files are saved one after another, the data is first, and the metadata follows.

    Iterating over TarReader yields tuples of ``(Optional[manifest], filename)`` where
    ``manifest`` is a Lhotse manifest with binary data attached to it, and ``filename``
    is the name of the data file inside tar archive.
    """

    def __init__(self, source: Pathlike) -> None:
        self.source = source

    def __iter__(
        self,
    ) -> Generator[Tuple[Optional[Manifest], Path], None, None]:
        with tarfile.open(fileobj=open_best(self.source, mode="rb"), mode="r|*") as tar:
            for ((data, data_path), (meta, meta_path)) in iterate_tarfile_pairwise(tar):
                if meta is not None:
                    meta = deserialize_item(decode_json_line(meta.decode("utf-8")))
                    fill_shar_placeholder(manifest=meta, data=data, tarpath=data_path)
                yield meta, data_path


def iterate_tarfile_pairwise(
    tar_file: tarfile.TarFile,
) -> Generator[Tuple[Optional[bytes], Optional[Manifest], Path, Path], None, None]:
    result = []
    for tarinfo in tar_file:
        if len(result) == 2:
            yield tuple(result)
            result = []
        result.append(parse_tarinfo(tarinfo, tar_file))

    if len(result) == 2:
        yield tuple(result)

    if len(result) == 1:
        raise RuntimeError(
            "Uneven number of files in the tarfile (expected to iterate pairs of binary data + JSON metadata."
        )


def parse_tarinfo(
    tarinfo: tarfile.TarInfo, tar_file: tarfile.TarFile
) -> Tuple[Optional[bytes], Path]:
    """
    Parse a tarinfo object and return the data it points to as well as the internal path.
    """
    path = Path(tarinfo.path)
    if path.suffix == ".nodata" or path.suffix == ".nometa":
        return None, path
    data = tar_file.extractfile(tarinfo).read()
    return data, path


class InMemoryTarIterator:
    """
    In-memory variant of TarIterator that loads the entire tar file into memory before iterating.

    This implementation solves NFS stale file handle issues by minimizing file handle lifetime:
    - File is opened, read completely, and closed within seconds (not minutes)
    - All subsequent processing happens from in-memory buffer
    - No long-lived file handles that can become stale

    **When to use this**:
    - Multi-worker dataloading (num_workers > 0)
    - Multi-node distributed training
    - NFS or network filesystem storage
    - When encountering "Stale file handle" errors

    **Trade-offs**:
    - Higher memory usage: ~shard_size per worker (e.g., 640MB - 2GB per worker)
    - Initial loading delay: 2-5 seconds before first item
    - Not suitable for extremely large tar files (> 5GB)

    **Usage**:
    Drop-in replacement for TarIterator - same interface, same output::

        # Original
        from lhotse.shar.readers.tar import TarIterator
        tar_iter = TarIterator("/path/to/shard.tar")

        # Replace with in-memory version
        from lhotse.shar.readers.tar import InMemoryTarIterator
        tar_iter = InMemoryTarIterator("/path/to/shard.tar")

    Or modify LazySharIterator to use this implementation::

        # In lhotse/shar/readers/lazy.py, replace:
        from lhotse.shar.readers.tar import TarIterator
        # with:
        from lhotse.shar.readers.tar import InMemoryTarIterator as TarIterator

    The interface is 100% compatible - no other code changes needed.
    """

    def __init__(self, source: Pathlike) -> None:
        """
        Initialize the iterator with a tar file path.

        Args:
            source: Path to the tar file (local path, NFS path, or URI)
        """
        self.source = source

    def __iter__(
        self,
    ) -> Generator[Tuple[Optional[Manifest], Path], None, None]:
        """
        Iterate over tar file members, yielding (manifest, filename) tuples.

        This method:
        1. Quickly reads entire tar file into memory (seconds)
        2. Closes the file handle immediately
        3. Processes tar members from in-memory buffer
        4. Yields items with identical format to TarIterator

        Yields:
            Tuple of (manifest, filename) where manifest is a Lhotse object
            (Recording, Features, Array, etc.) with binary data attached,
            and filename is the Path of the data file inside the tar archive.
        """
        # Step 1: Load entire tar file into memory
        # File handle is only open during this read (2-5 seconds for typical shard)
        import time
        max_retries = 3
        retry_delay = 1.0
    
        for attempt in range(max_retries):
            try:
                with open_best(self.source, mode="rb") as f:
                    tar_bytes = f.read()
                break  # 成功读取，退出重试循环
            except OSError as e:
                if e.errno == 116:  # Stale file handle
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                raise
        
        
        
        
        # with open_best(self.source, mode="rb") as f:
        #     tar_bytes = f.read()
        # File handle is now closed - no risk of stale handle

        # Step 2: Create tar file object from in-memory bytes
        tar_fileobj = io.BytesIO(tar_bytes)

        # Step 3: Process tar members from memory
        # Use mode="r" instead of "r|*" - allows random access since entire file is available
        with tarfile.open(fileobj=tar_fileobj, mode="r") as tar:
            # Reuse existing pairwise iteration logic
            for ((data, data_path), (meta, meta_path)) in iterate_tarfile_pairwise(tar):
                if meta is not None:
                    # Deserialize metadata and attach binary data
                    meta = deserialize_item(decode_json_line(meta.decode("utf-8")))
                    fill_shar_placeholder(manifest=meta, data=data, tarpath=data_path)
                yield meta, data_path
