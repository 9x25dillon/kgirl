"""Glyph. Nothing to see here."""

from __future__ import annotations

import base64
import re
import zlib

_SEED = (
    'c-'
    'pO3ZHntK4E(>RC<y(LkPl!G=>DgABiTvQY`PD(F*I!2?2JA~%H#3*X?dPEy|tguaJspf=TUca6;%;p!JXSDpbcpKd{l^h9@V}K+b'
    '%Hcm`Y*_mEEhrO|8E@CL;=kzlhxZ8rTP+?ZC^>P2ZTS#=~sV9h{b+`E4X@TFyX5Ds|cHcj+5P#Pc)40rWL&wm;0S$i!ZIq)Q=*H_'
    '*VTUlhQ~8@AkH^_Ij$B(v%pU1~)8GuVa94%;=$u-#K4Wi*dybUdZL)-acmHj?_<vo;Vg>+Y+P6me`tGmXGyekB=PMp-YSLU}C+P!'
    'TMSyUy)_vfVXFVXds1N-5}Q1^C@J*^^U5d{yC?iFGU5yvVPt{Va>I{8?e0mgj~^|NGD)A~IZ4?OKF0|0Zj?OBpMC;sVg6AiGsX=-'
    'S-uG*Ny<HF13_Utn`46Hl$6kH9+XX>3Bz{TOvq!q_K@pe6HXTQ&8=NET_%bRMoomJA%kw!i=xhXCeck-$0W1-'
    '7@!ube2vS|kJFM2nhv+KSq-IPI0=!8}NaKij&l3O4IzpXKm3Igkxp@8W1_FhldI*wRD9YC5uURE98ju4Z^dlaT*tSOQ9GghZ8|wc'
    '?-'
    '`KRi)tIwR7lrEX(vrCdW<r8Wj^wfHd;W5LIK@ZdnpnMpoNzx@CQp=>$>Ov5O=ad==hMVlNHy#>Q7agLrG{lF0!{F=a+LJ*4>DNpV'
    'L2ct(TGK!0id6WOx2Zz@k_dqBtO`!IY4&X^|#=_omy&66conwVau;nC+0s#J?N)Axvch?+Ttsr=ujH}E>1HN$794qB&DPYR=r^NG'
    '!?SJNvIr;W}@d;;0nI(6t65=Y|n`&38eh3jpKGGZInep5Dqa}WRq={ug8V@&4wfhNl3KtZTGJFZh<W;*(%8j6JIkt_@bt*96kfrL'
    '6o#H!=j#+{3&uOGNBR~cnkL1wU<nT+5QFE{uR%fMQh9e1)`G`Z^e#0@}k=W+TL5g9uAwm`O?7wd~6zYn>jmETe$j3Vf{K@`(Qhdc'
    '>`L3?@!k!&D)@Q2r<ULSlO<eV3RMobck0Mhpd6-UMBjK-Rl;<eIJ5E9oZA*kG#D9||=CdZQ`eDx(&4&PPu9SV#s>AuXcDHv5gEpg'
    'r2Z>87eZ7eTn>SLPd(@cI%noVkbIrerV<;9H9XS#0!~WbuD-*|a+wN0SWK`?Rq5F29xf0N!H=eQj2ahWaiv'
)

_KNOCKS = {"hello", "hello world", "print hello world", "printf hello world", "echo hello world"}


def knocks(argv) -> bool:
    words = re.sub(r"[^a-z]+", " ", " ".join(argv).lower()).split()
    return " ".join(words) in _KNOCKS


def reveal() -> str:
    return zlib.decompress(base64.b85decode("".join(_SEED))).decode("utf-8")
