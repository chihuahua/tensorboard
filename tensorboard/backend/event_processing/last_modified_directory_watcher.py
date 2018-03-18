# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the implementation for the LastModifiedDirectoryWatcher class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import os

import tensorflow as tf


from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper

_DEFAULT_MAX_EVENT_FILES_TO_TRACK = 5


class EventsFileEntry(object):
  """Encapsulates resources related to an events file."""
  def __init__(self, path, modified_time, loader_factory):
    self.path = path

    # The last modified time (in nanoseconds) of the file when events were last
    # read from the file.
    self.mtime_at_last_read = modified_time

    # The last modified time of the file last recorded. Note that
    # this property differs from `mtime_at_last_read` because
    self.last_recorded_mtime = modified_time

    # The loader used to read events from this file. If this events file is no
    # longer being tracked, this property is None.
    self.loader = loader_factory(path)


class LastModifiedDirectoryWatcher(directory_watcher.DirectoryWatcher):
  """Loads events based on which events files were last modified.

  Unlike the single file directory watcher, this directory watcher continuously
  yields events from the last modified events file. Hence, this watcher supports
  multiple writers concurrently writing to the same run (via different events
  files).
  """

  def __init__(self,
               directory,
               loader_factory,
               path_filter=lambda x: True,
               max_events_files_to_track=None):
    """Constructs a new SingleFileDirectoryWatcher.

    Args:
      directory: The directory to load files from.
      loader_factory: A factory for creating loaders. The factory should take a
        path and return an object that has a Load method returning an
        iterator that will yield all events that have not been yielded yet.
      path_filter: If specified, only paths matching this filter are loaded.
      max_events_files_to_track: The max number of events files to track. If an
        events file is tracked, TensorBoard may return to read events from it
        after moving on to reading a different events file. We limit the number
        of events files tracked in order to conserve file handlers. Defaults to
        _DEFAULT_MAX_EVENT_FILES_TO_TRACK.

    Raises:
      ValueError: If path_provider or loader_factory are None.
    """
    if directory is None:
      raise ValueError('A directory is required')
    if loader_factory is None:
      raise ValueError('A loader factory is required')
    self._directory = directory
    self._loader_factory = loader_factory
    self._path_filter = path_filter

    # Maps path (string) to events file entry for a tracked file.
    self._path_to_events_file_entry = {}
    self._max_events_files_to_track = (
        max_events_files_to_track or _DEFAULT_MAX_EVENT_FILES_TO_TRACK)

    # The number of events files currently being tracked.
    self._events_files_tracking_count = 0

  def Load(self):
    """Loads new values."""
    try:
      for event in self._LoadInternal():
        yield event
    except tf.errors.OpError:
      if not tf.gfile.Exists(self._directory):
        raise DirectoryDeletedError(
            'Directory %s has been permanently deleted' % self._directory)

  def _LoadInternal(self):
    """Internal implementation of Load().

    The only difference between this and Load() is that the latter will throw
    DirectoryDeletedError on I/O errors if it thinks that the directory has been
    permanently deleted.

    Yields:
      All values that have not been yielded yet.
    """
    while True:
      paths = [path
               for path in io_wrapper.ListDirectoryAbsolute(self._directory)
               if self._path_filter(path)]
      if not paths:
        tf.logging.warn(
            'No event files found within directory %r.', self._directory)
        raise StopIteration

      if not self._path_to_events_file_entry:
        # No file has been read yet. Start with the first one in lexicographical
        # order.
        chosen_entry = self._CreateNewEntry(min(paths), 0)
      else:
        chosen_entry = None

        # TODO(chihuahua): Ensure this operation works on Google Cloud Storage.
        # TODO(chihuahua): Add the BulkStatWithException method to TensorFlow's
        # gfile module. 
        stats = [tf.gfile.Stat(path) for path in paths]

        # Sort files by last modified time, then by file name.
        sorted_stats = [(stat.mtime_nsec, path, stat)
                        for (stat, path) in zip(stats, paths)]
        sorted_stats.sort()

        # Update last recorded mtimes for tracked files. This should be done 
        # before we untrack any files because we decide to untrack based on the
        # last recorded mtime.
        for (_, path, stat) in sorted_stats:
          if path not in self._path_to_events_file_entry:
            continue
          self._path_to_events_file_entry[path].last_recorded_mtime = (
              stat.mtime_nsec)

        for (_, path, stat) in sorted_stats:
          if path not in self._path_to_events_file_entry:
            # We have not read from this file yet. Definitely do so. Maybe 
            # TensorBoard is starting up.
            chosen_entry = self._CreateNewEntry(path, stat.mtime_nsec)
            break

          entry = self._path_to_events_file_entry[path]
          if not entry.loader:
            # We no longer track this file. It will never be read from again.
            if stat.mtime_nsec > entry.mtime_at_last_read:
              # However, the file has nonetheless been modified recently. Warn the
              # user.
              tf.logging.warn(
                  ('The events file %r has been modified after it has been '
                  'untracked. TensorBoard will no longer read this events '
                  'file, so any new events will not be shown in TensorBoard.'),
                  path)
            continue
          
          if stat.mtime_nsec > entry.mtime_at_last_read:
            # This events file has been modified since the last time TensorBoard
            # resumed reading events from it. Go back to this file and read
            # events from it even though we have already previously read from
            # this file. It might have new events written to it.
            chosen_entry = self._path_to_events_file_entry[path]
            # Updating the last modified time at which we resumed reading from
            # the events file.
            chosen_entry.mtime_at_last_read = stat.mtime_nsec
            break
        
        if chosen_entry is None:
          tf.logging.info(
              'No events files found to be updated within directory %r.',
              self._directory)
          raise StopIteration

      # Keep reading events from the chosen file.
      for event in chosen_entry.loader.Load():
        yield event

  def _CreateNewEntry(self, path, mtime_nsec):
    """Creates a new file entry. Appropriately updates internal state.
    
    Args:
      path: The path of the file.
      mtime_nsec: The last modified time of the file.

    Returns:
      A newly created events file entry.
    """
    entry = EventsFileEntry(path, mtime_nsec, self._loader_factory)
    if self._events_files_tracking_count >= self._max_events_files_to_track:
      # We are already tracking the max number of files. Stop tracking the least
      # recently modified file that is currently being tracked before tracking
      # this new one.
      path_to_remove = None
      min_mtime = None
      for (previous_path, previous_entry) in self._path_to_events_file_entry:
        if not previous_entry.loader:
          # This file had been previously untracked.
          continue

        if path_to_remove is None:
          # Start comparisons with some initial entry.
          path_to_remove = previous_path
          min_mtime = previous_entry.last_recorded_mtime
          continue

        if ((previous_entry.last_recorded_mtime, previous_path) <
            (min_mtime, path_to_remove)):
          # A new minimum. Possibly stop tracking this entry.
          min_mtime = previous_entry.last_recorded_mtime
          path_to_remove = previous_path

      # Stop tracking the entry. We preserve the entry within the mapping to
      # remember how this file is untracked. Discard the file loader. We will
      # never again read from this file. We rely on python to garbage-collects
      # the file handler. The loader lacks a method for closing.
      self._path_to_events_file_entry[path_to_remove].loader = None
    else:
      self._events_files_tracking_count += 1

    # Store this new entry.
    self._path_to_events_file_entry[path] = entry
    return entry
