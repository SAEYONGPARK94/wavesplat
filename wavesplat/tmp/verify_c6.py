import json
d = json.load(open('tmp/record_wavelength.json'))
print(f"{'rec':7s}{'reg':>4}{'patch':>14}{'lam_T':>8}{'patch/lam':>10}{'r_fan':>7}{'r/lam':>7}  note")
for r in d:
    note = ''
    if not r['registered']:
        note = f"fan SHRUNK to r={r['r_fan']:.1f} m (registered fans: 40 / 100 m)"
    print(f"{r['name']:7s}{'Y' if r['registered'] else '-':>4}"
          f"{r['Lx']:6.0f}x{r['Ly']:<7.0f}{r['lam_temporal']:8.1f}"
          f"{r['L_over_lam']:10.2f}{r['r_fan']:7.1f}{r['aperture_ratio']:7.3f}  {note}")

reg = [r for r in d if r['registered']]
print()
print("registered fan r/lam span : %.3f - %.3f" % (
    min(x['aperture_ratio'] for x in reg), max(x['aperture_ratio'] for x in reg)))
print("registered patch/lam span : %.2f - %.2f" % (
    min(x['L_over_lam'] for x in reg), max(x['L_over_lam'] for x in reg)))
print()
print("patch/lam of the candidates (the GT field itself spans this many wavelengths):")
for r in d:
    if not r['registered']:
        flag = '  <-- SUB-WAVELENGTH GT' if r['L_over_lam'] < 1.0 else ''
        print(f"   {r['name']:7s} patch/lam = {r['L_over_lam']:.2f}{flag}")
