import numpy as np
import copy

def find_rebinning(hist, bins, bin_threshold=10, bin_uncert_frac_threshold=0.15, uncert=None):

  if isinstance(hist, list):
    hist = np.array(hist)
  if isinstance(uncert, list):
    uncert = np.array(uncert)
  if isinstance(bins, list):
    bins = np.array(bins)

  inc_uncert = False
  if uncert is None:
    uncert = np.sqrt(hist)
    inc_uncert = True

  # rebin left to right
  loop_ind = 0
  while loop_ind < 1000:
    loop_ind += 1
    for ind in range(len(bins)-2):
      bin_content = hist[ind]
      bin_uncert_frac = uncert[ind] / bin_content if bin_content > 0 else np.inf
      if (bin_content < bin_threshold) or (bin_uncert_frac > bin_uncert_frac_threshold):
        old_bins = copy.deepcopy(bins)
        bins = np.concatenate((bins[:ind+1], bins[ind+2:]))
        hist, uncert = rebin_histogram(hist, old_bins, bins, uncert)
        break
    if ind == len(bins) - 3:
      break

  # rebin right to left
  loop_ind = 0
  while loop_ind < 1000:
    loop_ind += 1
    for ind in range(len(bins)-2, 1, -1):
      bin_content = hist[ind]
      bin_uncert_frac = uncert[ind] / bin_content if bin_content > 0 else np.inf
      if (bin_content < bin_threshold) or (bin_uncert_frac > bin_uncert_frac_threshold):
        old_bins = copy.deepcopy(bins)
        bins = np.concatenate((bins[:ind], bins[ind+1:]))
        hist, uncert = rebin_histogram(hist, old_bins, bins, uncert)
        break
    if ind == 1:
      break

  return bins


def rebin_histogram(hist, bins, rebin_bins, uncert=None):

  if isinstance(hist, list):
    hist = np.array(hist)
  if isinstance(uncert, list):
    uncert = np.array(uncert)
  if isinstance(bins, list):
    bins = np.array(bins)
  if isinstance(rebin_bins, list):
    rebin_bins = np.array(rebin_bins)

  # check all rebin_hist in bins
  for b in rebin_bins:
    if b not in bins:
      raise ValueError(f"Rebin bin {b} not found in original bins")

  # check bins and rebin_bins are in ascending order
  if not np.all(np.diff(bins) > 0):
    raise ValueError("Original bins are not in ascending order")
  if not np.all(np.diff(rebin_bins) > 0):
    raise ValueError("Rebin bins are not in ascending order")

  # Set up new objects
  rebin_hist = np.zeros(len(rebin_bins) - 1)
  if uncert is not None:
    rebin_uncert = np.zeros(len(rebin_bins) - 1)

  # Rebin
  for ind in range(len(bins)-1):
    bin_ind = None
    for r_ind in range(len(rebin_bins)-1):
      if bins[ind] >= rebin_bins[r_ind] and bins[ind] < rebin_bins[r_ind+1]:
        bin_ind = r_ind
        break

    if bin_ind is None:
      raise ValueError(f"Bin {bins[ind]} not found in rebin bins {rebin_bins}")

    rebin_hist[bin_ind] += hist[ind]
    if uncert is not None:
      rebin_uncert[bin_ind] = (rebin_uncert[bin_ind]**2 + uncert[ind]**2)**0.5

  # Return
  if uncert is not None:
    return rebin_hist, rebin_uncert
  return rebin_hist

