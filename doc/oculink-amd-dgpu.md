# The R9700 does not enumerate over OCuLink on this MS-S1 Max

A firmware investigation on **`halo-win`**, traced from "the card is not in Device
Manager" down to one byte in one UEFI variable — and then to the discovery that
the byte cannot be written from any channel a user has.

**Status: diagnosed, not fixed, and not the whole story.** The suspected setting
could never be changed, so the diagnosis is well-supported but **unverified** —
only Minisforum can close that part. Since it was written, one owner has reported
an R9700 working over OCuLink on an MS-S1 Max with a *DEG1* dock, which the
theory here does not predict; see [What other machines show](#what-other-machines-show)
for what that does and does not overturn.

## Why this is in a multi-GPU repo

`halo-win` reaches its second GPU over a Thunderbolt 5 dock, and that link is
unreliable: eight of twelve dual-GPU loads fail with `unspecified launch failure`
(see [benchmarks.md](benchmarks.md#the-dual-layout-is-not-stable-on-this-dock-and-the-link-rate-is-not-the-variable)).
OCuLink is a direct PCIe connection with none of that tunnelling, so moving the
R9700 onto it would likely make the dual layout reliable. It does not work, and
this file is why.

**The reason to want it is not throughput.** Measured on this rig with the same
lane count, the tunnel costs 3% of prefill when only activations cross it and
~13% when expert weights do
([what Thunderbolt 5 costs](benchmarks.md#what-thunderbolt-5-costs-against-oculink)).
What OCuLink would buy is a layout that loads every time and a card that idles
down — see the last paragraph of [What other machines show](#what-other-machines-show).

NVIDIA cards work on the identical path. An RTX 5090 and an RTX PRO 6000 both
enumerated over OCuLink on this machine; the R9700 does not. That asymmetry is
the thread this whole investigation pulls on.

## The symptom

With the R9700 attached over OCuLink (RIITOP PCIe-to-OCuLink adapter, Minisforum
DEG2 dock), the card powers up, its fan spins, the machine boots, and Windows
does not list the GPU anywhere.

On another MS-S1 Max, a user running Linux reported the sharper version of the
same symptom: the **root port itself disappears** from PCI enumeration, the tree
skipping `00:03.0` straight to `00:03.2`, while a GTX 1080 on the identical path
enumerates and maps to `00:03.1`
([r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1wi87ac/mss1_max_radeon_ai_pro_r9700_over_oculink_no/)).

> That root-port observation is **not from this rig** — it comes from that user's
> Linux logs. It has not been reproduced here, because `halo-win` runs Windows
> and the equivalent check needs a live Linux boot with the card attached. It is
> included because it explains the mechanism, not because it was measured here.

A vanished root port is the important part. If the card, cable or link training
were at fault you would still see the port, empty or erroring. A root port is
silicon inside the SoC; it cannot go missing. Its absence means firmware
deliberately disabled and hid it during POST. That is also why `pci=realloc` and
a PCI rescan do nothing — there is no bridge left for the OS to work with.

## The finding

BIOS SHWSA 1.11 contains an AMD PBS (Platform BIOS Setup) option that the retail
setup UI does not expose:

```
Setup Question = Non-Eval Discrete GPU Support
Help String    = Enable to support Non-Eval Discrete GPU that doesn't have
                 specific EVAL_PWRGD(B30), EVAL_PRESENT#(A5)
Variable       = AMD_PBS_SETUP  (GUID A339D746-F678-49B3-9FC7-54CE0F9DF226)
Offset         = 0x35
Current value  = 0x00  (Disabled)
```

The AMD reference design for Strix Halo expects an AMD discrete GPU to sit in an
**EVAL slot**, which carries two sideband pins beyond the PCIe signals:
`EVAL_PWRGD` (B30, "card power is good") and `EVAL_PRESENT#` (A5, "card is
present"). A retail card reached through an OCuLink adapter has no way to drive
either. The firmware enters its AMD-dGPU path, waits for a presence signal that
never arrives, concludes the card is absent, and disables the port.

NVIDIA cards never enter that path — PBS has a separate `NVIDIA DGPU Power
Enable` option for them, and it is likewise disabled — so they fall through to
generic PCIe handling and work.

The surrounding PBS options confirm the shape of it: `EVAL Slot Power Enable`,
`Special Display Features = HybridGraphics`, `Discrete GPU HPD Circuitry`,
`Discrete GPU's SSID/SVID`, `D3Cold Force Gen1`. This is laptop hybrid-graphics
machinery, and a retail card on a dock is not what it was written for.

## What other machines show

Two reports from the [r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1wi87ac/mss1_max_radeon_ai_pro_r9700_over_oculink_no/)
bear directly on the theory above. One supports it, one does not fit it.

### Framework Desktop enumerates an R9700 — with two caveats

A Framework Desktop owner posted the allocation itself rather than a report of
it working:

```
c3:00.0 VGA compatible controller: [AMD/ATI] Navi 48 [Radeon AI PRO R9700] (rev c0)
Subsystem: Framework Computer Inc. Device 000a
Region 0: Memory at 2800000000 (64-bit, prefetchable) [size=32G]
Region 2: Memory at 3000000000 (64-bit, prefetchable) [size=256M]
Kernel driver in use: amdgpu
```

A 32 GiB prefetchable BAR0 allocated and `amdgpu` bound, on the same Ryzen AI
Max+ 395 silicon. **The limit is not in the CPU** — that is what this capture
settles, and it is the strongest evidence against a silicon explanation.

Two things it does not settle, both worth stating before leaning on it:

- **`Subsystem: Framework Computer Inc. Device 000a`** is a Framework-branded
  R9700, not a retail card. PBS carries a `Discrete GPU's SSID/SVID` option, so
  a firmware path keyed on subsystem ID is not excluded. What was demonstrated
  is that *their* card works, which is weaker than that any R9700 does.
- **The topology is not matched.** The reported tree is
  `00:02.5-[c1-c3]----00.0-[c2]----00.0-[c3]--+-00.0 Navi 48` — two bridges
  between root port and GPU, on root port `02.5`. `halo-win` reaches its x4 slot
  through `00:03.1` with no intermediate bridge. Same silicon, same class of
  card, different path.

### One MS-S1 Max is reported working, and the theory does not predict it

Another owner in that thread runs an R9700 over OCuLink on an MS-S1 Max, through
a **DEG1** dock and the same RIITOP adapter used here. It took three things:

- the link would not train at Gen4, so **Gen3 had to be forced in BIOS** before
  the card appeared in `lspci` at all
- **`amdgpu.runpm=0`**, because the card could not power back up from idle over
  OCuLink
- **three DIP switches** under a panel on the underside of the DEG1, all moved
  off their factory positions

This matters more than anything else on this page: the EVAL-pin theory predicts
that no retail AMD card can work here, and one does. Three differences keep the
theory alive rather than refuting it, and none of them is comfortable.

1. **The failure modes are not the same.** That machine's root port was present
   and failing to train. Here, and for the thread's author, the root port is
   **gone** — `00:03.1` absent from the tree entirely. A port that trains badly
   and a port that was never published are different faults.
2. **Gen3 does not help here.** Forced Gen3 and Gen4 were both tested on
   `halo-win`; neither changes anything. The thread's author also tried the DEG1
   switches, with no effect.
3. **DEG1 against DEG2.** This rig uses a DEG2, the newer dock, which also
   carries a USB4/Thunderbolt path the DEG1 does not. Whether that changes how it
   routes OCuLink is untested, and it is the most interesting variable left. The
   DIP switches are part of the same question: nothing equivalent has been located
   on the DEG2.

So "AMD dGPUs do not enumerate over OCuLink on this platform" is too strong as a
general claim. It holds on this rig and on at least one other machine, and fails
on at least one machine with a different dock. What none of that changes is the
argument for the default: **it should not require forcing Gen3 and moving
undocumented DIP switches** to use a supported card on a supported port.

One detail in that working report lines up with something measured here. That
card could not return from idle over OCuLink — and on `halo-win` the R9700 over
Thunderbolt draws **40–50 W doing nothing**, against **under 10 W for the whole
dock including the GPU** when the same card and the same DEG2 run over OCuLink
on an Intel desktop. No driver version or Windows power setting reached it. The
same power-state machinery appears to be implicated in both, on a path where the
GPU never enters its low-power state.

## Every write channel is blocked

The option is reachable and visibly toggleable. It still cannot be changed.

| Channel | Result |
|---|---|
| `setup_var.efi` from a UEFI shell | `WRITE_PROTECTED` for both `Setup` and `AMD_PBS_SETUP` |
| `SetFirmwareEnvironmentVariableW` from Windows | `ERROR_WRITE_PROTECT` (19) |
| AMISCE 5.05.01.0002 script import | `Cannot update protected variable`, then `Admin password does not exist` (86) with a password set |
| AMISCE single-variable / single-question update | same password error |
| Smokeless_UMAF form browser | `Submit Fail For Form: AMD PBS Option` |
| **SREP + the firmware's own setup browser** | **accepts the change, does not survive reboot** |

The last row is the decisive one. After unhiding the menu with
SmokelessRuntimeEFIPatcher, the option is set in AMI's own setup UI and saved
with F10 — the one path that is supposed to have write authority. After a
reboot, `AMD_PBS_SETUP` reads back byte-identical to before.

A control test settles what that means. `Wireless LAN Recovery` (offset `0x3F`),
a harmless non-graphics option in the same variable, was changed the same way.
It did not persist either. **Nothing in `AMD_PBS_SETUP` persists**, so this is
not the graphics options being singled out — the whole variable is immutable
from userland.

All 15 graphics-related offsets also read exactly their IFR defaults, which is
consistent with `AmdPbsSetupDxe` rewriting the variable from defaults on every
boot. That is a hypothesis, not a measurement: it fits the evidence but was not
proven directly.

Windows cannot even read the AMI `Setup` variable at runtime — it returns Win32
error 203, being stored boot-services-only. `AMD_PBS_SETUP` reads fine; it is
writes that fail.

One module names itself as a suspect: `AmiSetupNVLockDxe`
(GUID `A51BC7A4-0ED6-44C2-B5FB-B86FDE077DE1`), a 6 KB DXE driver whose only two
strings are `NvLockMailbox` and `Setup`. It locks the AMI `Setup` variable.
Nothing was found that explicitly locks `AMD_PBS_SETUP`, which is part of why
the rewrite-from-defaults theory is the leading one.

## Reproducing the analysis

Extract the setup forms from the BIOS image. `SHWSA.BIN` ships inside
Minisforum's BIOS update package:

```bash
# UEFIExtract from LongSoft/UEFITool, ifrextractor from LongSoft/IFRExtractor-RS
UEFIExtract.exe SHWSA.BIN report

# AMD PBS forms (Non-Eval lives here)
UEFIExtract.exe SHWSA.BIN BBB77CB9-762D-436C-AC40-8EE4901C3446 -o pbs -m body -t FF
ifrextractor.exe pbs/body_1.bin

# the main AMI Setup forms (the hidden Advanced tree lives here)
UEFIExtract.exe SHWSA.BIN 899407D7-99FE-43D8-9A21-79EC328CAC21 -o setup -m body -t FF
ifrextractor.exe setup/body_1.bin
```

Read the live values from Windows with `GetFirmwareEnvironmentVariableW` on
`AMD_PBS_SETUP` / `{A339D746-F678-49B3-9FC7-54CE0F9DF226}`, elevated, after
enabling privilege 22 (`SeSystemEnvironmentPrivilege`) via `RtlAdjustPrivilege`.

## Unhiding the menu, for whoever needs it

The vendor UI suppresses the AMI `Advanced` tab outright. The AMD PBS pages sit
behind it, gated on `Setup[0xAB] == 0x5A` — a vendor unlock byte that is itself
write-protected. [SmokelessRuntimeEFIPatcher](https://github.com/hboyd2003/SmokelessRuntimeEFIPatcher)
(the [0.2.x fork](https://github.com/Maxinator500/SmokelessRuntimeEFIPatcher-RUS)
adds the op used here) cancels the suppression in RAM:

```
Op SuppressIfPatcher

Op LoadFromFV
AMITSE
Op Exec
End
```

Boot a UEFI Shell from USB with `SmokelessRuntimeEFIPatcher.efi` and one `.cfg`
in the root — the fork loads the first `.cfg` it finds, so keep exactly one.
`_ENG` in the filename switches messages to English, `_LOG` writes `SREP.log`.
After it reports success, `exit` drops into the BIOS with
`____ Crb Advanced ____` present, and AMD PBS under it.

This patches RAM only. Nothing is written to flash, and a reboot undoes it, so
the menu has to be unhidden again for every attempt. Unsuppressing also reveals
CRB and debug pages that are hidden for good reasons — change nothing else.

The alternative, if `SuppressIfPatcher` is unavailable, is patching the gate
comparison directly. The condition encodes as `12 86 3A 01 5A 00`
(`EqIdVal QuestionId 0x13A, Value 0x5A`); replacing `5A 00` with `00 00` turns
`NOT(Setup[0xAB] == 0x5A)` into `NOT(0x00 == 0x00)`, which is false, so the
items show. There are 11 occurrences in the Setup module.

## What is left

Nothing, on this side of the firmware. The remaining technical option is editing
the BIOS image so `AMD_PBS_SETUP[0x35]` defaults to `0x01` and flashing it with
the `AfuEfix64.efi` that ships in the update package. On an AMD platform parts of
the image are PSP-signed, so a modified image may simply be refused — that is
brick risk with an uncertain payoff, and it is not recommended while the vendor
has not answered.

What Minisforum would need to do, in order of preference:

1. Ship `Non-Eval Discrete GPU Support = Enabled` as the default.
2. Expose the AMD PBS / CRB Advanced menus, and unhide `Above 4G Decoding`
   (`Setup` offset `0x65`, currently wrapped in `SuppressIf TRUE` although its
   default is already Enabled).
3. Make `AMD_PBS_SETUP` writable from the firmware's own setup browser. Today a
   user can see the option, toggle it, and save it, and it silently reverts.
4. Update AMD PI from 1002B patchC (their 1.09 release notes) to 1.0.0.2c, the
   level Framework Desktop ships.

> **Request 4 is the weak one, and it should be made honestly.** 1.0.0.2c is
> traceable: Framework Desktop BIOS 3.04 moved AMD PI to 1.0.0.2, and 3.06
> (stable, 2026-07-28) to 1.0.0.2c. But that changelog has ten items and **none
> of them mentions discrete GPUs, PCIe enumeration or graphics**, beyond a boot
> hang with "specific video encoder cards" and an iGPU memory cap. No Framework
> release note claims a dGPU fix, so pairing the version with the working
> enumeration is correlation, not a documented cause. `1002B patchC` may also
> parse as 1.0.0.2B patch C rather than something strictly older than 1.0.0.2C,
> which invites the reply "we already ship that".
>
> The claim that does not depend on any of this: **the same option is exposed to
> end users on other AMD platforms.** The ASUS ROG STRIX X670E BIOS manual
> documents `AMD PBS → Graphics Features → Non-Eval Discrete GPU Support
> [Disabled] / [Enabled]` with the identical EVAL-pin help string. It is a
> normal, shippable, user-facing PBS item — which is what request 2 asks for.

Tracking threads:
[r/MINISFORUM](https://www.reddit.com/r/MINISFORUM/comments/1wm94cz/mss1_max_doesnt_work_with_amd_gpus_radeon_ai_pro/),
[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1wi87ac/mss1_max_radeon_ai_pro_r9700_over_oculink_no/).

## What was ruled out along the way

Each of these was tested and is not the cause:

- **Resizable BAR**, on and off. (It does matter for NVIDIA here: the RTX PRO
  6000 needed ReBAR **off** to enumerate over OCuLink, because OCuLink on this
  machine requires Resizable BAR disabled at all — see [systems.md](systems.md).)
- **IOMMU** disabled. A fix reported by another MS-S1 owner; it was active
  during several of these tests and changed nothing.
- **Above 4G Decoding**. Hidden from the UI, but its stored default is already
  Enabled.
- **`Above 4GB MMIO Limit`**, which reads `40bit (1TB)` — MMIO space is not the
  constraint.
- **Link speed forced to Gen3**, and Gen4. Gen3 is what made the difference on
  the one MS-S1 Max reported working; here it changes nothing.
- **A BIOS predating 1.11.** Another owner points at 1.05 for a PCIe enumeration
  fix; this rig is on 1.11, which postdates it.
- **Cables, adapters, docks, PSU wiring, and OS** — covered across the two
  Reddit threads with three adapters, two docks, two Linux distributions and
  Windows. The machine here uses Minisforum's own DEG2 dock and cable, so a
  marginal third-party cable is excluded.
- **BIOS administrator password**, set and confirmed to prompt at POST. AMISCE
  still reports it does not exist, so that tool's password detection is
  incompatible with this 2026 firmware.
