; Import the Python module using the IDL Python bridge
models = Python.Import('mcalf.models')

; Restore the SAV files containing the specified variables
RESTORE, 'wavelengths_original.sav'  ; WAVELENGTHS
RESTORE, 'prefilter.sav'  ; PREFILT8542_REF_MAIN, PREFILT8542_REF_WVSCL
RESTORE, 'groundtruths.sav'  ; LABELS, LABELED_DATA
RESTORE, 'background.sav'  ; BG

; Initialise the model object
M = models.IBIS8542Model(original_wavelengths=WAVELENGTHS, prefilter_ref_main=PREFILT8542_REF_MAIN, prefilter_ref_wvscl=PREFILT8542_REF_WVSCL)

; Train and test the neural network
M->TRAIN(LABELED_DATA[*, 0:99], LABELS[0:99])
M->TEST(LABELED_DATA[*, 100:199], LABELS[100:199])

; Load the spectra into the model
RAW = READFITS('IBIS_scan_00100.fits')  ; You may wish to mask this array
M->LOAD_ARRAY(RAW, ['wavelength', 'row', 'column'])  ; Note: these dimensions are in Python array order, not the IDL order (i.e. reversed)

; Load the background intensities
M->LOAD_BACKGROUND(BG, ['row', 'column'])

; Example: classify the field of view
CLASSIFICATIONS_MAP = M->CLASSIFY_SPECTRA(ROW=INDGEN(1000), COLUMN=INDGEN(1000))

; Example: fit some spectra
FITS = M->FIT(ROW=600, COLUMN=600)
FIT1 = FITS[0]
FITS = M->FIT(ROW=[500, 501], COLUMN=500)
FIT2 = FITS[0]
FIT3 = FITS[1]
