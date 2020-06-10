from pathlib import Path

def full_filename(fname, outdir, ext='.csv', suffix=''):
    if suffix:
        res = str((Path(outdir) / fname).resolve()) + "_" + suffix + ext
    else:
        res = str((Path(outdir) / fname).resolve())  + ext
    return res

def clean_outdir(outdir):
    print("deleting all files in {dir}".format(dir=outdir))
    for f in Path(outdir).rglob('*'):
        f.unlink()

def logscale(x0, c, n):
    """logscale"""
    return [x0 * (c**i) for i in range(n)] 
