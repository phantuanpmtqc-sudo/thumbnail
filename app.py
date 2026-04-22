"""
app.py — Thumbnail Builder Pro v10
Dark theme · Dynamic pill width · Tracking -28 · Khớp PS
═══════════════════════════════════════════════════════════
Nâng cấp từ v9:
  - Tracking -28 (PS VA=-28, vẽ từng ký tự)
  - Font size 32.0 (PS 24pt × 96/72)
"""
from __future__ import annotations
import io, json, time, zipfile
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError
from image_processor import (
    CANVAS_SIZE, ThumbnailConfig, build_thumbnail,
    pil_to_bytes, sanitize_filename, list_available_fonts, measure_text_width,
)
from auth import require_login, logout_btn

# ═══ HẰNG SỐ (khớp PS) ═══
SHADOW_X, SHADOW_Y, SHADOW_BLUR, SHADOW_OP = 3, 4, 0, 60
FONT_WEIGHT = 700; WHITE_TOL = 18; TEXT_PADDING = 12
DEFAULT_FONT_FAMILY = "MontserratAlternates-Black"
DEFAULT_FONT_SIZE = 31.5       # PS 24pt × (96/72) = 32px
DEFAULT_TRACKING = 19.5        # ĐÃ SỬA THÀNH SỐ DƯƠNG ĐỂ CHỮ GIÃN THOÁNG RA
PILL_LEFT, PILL_HEIGHT, PILL1_TOP, PILL2_GAP = 20, 49, 14, 14
TEXT_Y_NUDGE = 1              # <--- CHỖ NÀY ĐỂ BẠN TỰ CHỈNH TEXT LÊN XUỐNG NÈ (-2 LÀ ĐẨY LÊN 2PX)
MAX_PILL_RIGHT = 520
MAX_UPLOAD_DIM, MAX_UPLOAD_MB, MIN_SRC_DIM = 1600, 20, 400
SUPPORTED_SIZES = [300, 600, 800, 1000, 1200]
APP_VERSION = "10.0"

def _fallback_bg(sz=600):
    bg = Image.new("RGBA",(sz,sz),(255,255,255,255))
    a = np.array(bg,dtype=np.float32); h = int(sz*.78)
    for c in range(3): a[h:,:,c] *= np.linspace(1,.93,sz-h)[:,None]
    bg = Image.fromarray(np.clip(a,0,255).astype(np.uint8),"RGBA")
    s = Image.new("RGBA",(sz,sz),(0,0,0,0))
    ImageDraw.Draw(s).ellipse([sz//2-210,int(sz*.82)-24,sz//2+210,int(sz*.82)+24],fill=(0,0,0,35))
    bg.alpha_composite(s.filter(ImageFilter.GaussianBlur(8))); return bg

# ═══ PAGE ═══
st.set_page_config(page_title="Thumbnail Builder Pro", page_icon="🖼️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container{padding-top:3.5rem !important;max-width:1500px;}
  #MainMenu,footer{visibility:hidden;}

  /* Dark enhancements */
  .stButton>button{border-radius:10px !important;font-weight:600 !important;}
  .stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#7c3aed,#8b5cf6) !important;border:none !important;}
  .stDownloadButton>button{
    border-radius:10px !important;font-weight:700 !important;
    background:linear-gradient(135deg,#10b981,#059669) !important;
    color:#fff !important;border:none !important;}

  .stTabs [data-baseweb="tab-list"]{gap:4px;}
  .stTabs [data-baseweb="tab"]{font-weight:700;font-size:13px;padding:10px 18px;}

  [data-testid="stMetricValue"]{font-size:26px !important;font-weight:800 !important;}
  [data-testid="stFileUploader"] section{
    border:2px dashed #4c1d95 !important;border-radius:12px !important;}
</style>
""", unsafe_allow_html=True)

if not require_login(): st.stop()

# ═══ PATHS ═══
_DIR = Path(__file__).resolve().parent
_ASSETS = _DIR / "assets"

@st.cache_resource(show_spinner=False)
def _bg():
    p = _ASSETS / "background.png"
    if p.exists():
        try: return Image.open(p).convert("RGBA")
        except: pass
    return _fallback_bg()

# ═══ PRESETS ═══
PRESETS = {
    "🎯 Chuẩn (PS mẫu)": dict(top_margin=155,bottom_margin=35,side_padding=40,product_scale=1.0,center_mode="centroid",font_size=31.5),
    "🍳 Đồ gia dụng":    dict(top_margin=170,bottom_margin=50,side_padding=50,product_scale=1.1,center_mode="centroid",font_size=32.0),
    "🎧 Điện tử":        dict(top_margin=160,bottom_margin=60,side_padding=40,product_scale=1.0,center_mode="centroid",font_size=32.0),
    "🧴 Mỹ phẩm":        dict(top_margin=155,bottom_margin=55,side_padding=60,product_scale=0.95,center_mode="bbox",font_size=32.0),
    "👕 Thời trang":      dict(top_margin=155,bottom_margin=45,side_padding=35,product_scale=1.05,center_mode="bbox",font_size=32.0),
    "📱 ĐT dọc":          dict(top_margin=160,bottom_margin=50,side_padding=70,product_scale=1.0,center_mode="bbox",font_size=32.0),
}

def _init():
    st.session_state.setdefault("preset_name","🎯 Chuẩn (PS mẫu)")
    st.session_state.setdefault("overrides",{})
    for k,v in PRESETS[st.session_state["preset_name"]].items():
        st.session_state.setdefault(f"cfg_{k}",v)
_init()

def _apply_preset(n):
    for k,v in PRESETS.get(n,{}).items(): st.session_state[f"cfg_{k}"]=v
    st.session_state["preset_name"]=n

# ═══ HELPERS ═══
def _compress(data, max_dim=MAX_UPLOAD_DIM):
    try:
        img=Image.open(io.BytesIO(data)); img.load()
    except: return data
    w,h=img.size
    if max(w,h)<=max_dim: return data
    s=max_dim/max(w,h); img=img.resize((int(w*s),int(h*s)),Image.LANCZOS)
    buf=io.BytesIO()
    if img.mode in ("RGBA","LA"): img.save(buf,"PNG",optimize=True)
    else: img.convert("RGB").save(buf,"JPEG",quality=95,optimize=True)
    return buf.getvalue()

def _read_excel(file):
    n=getattr(file,"name","").lower()
    df=(pd.read_csv(file,dtype=str,keep_default_na=False) if n.endswith(".csv")
        else pd.read_excel(file,dtype=str,keep_default_na=False))
    df.columns=[str(c).strip().lower().replace(" ","") for c in df.columns]
    m={}
    for c in df.columns:
        if c in ("id","sku","masp","ma","productid","code"): m[c]="id"
        elif c in ("text1","tt1","dong1","line1","title1"): m[c]="text1"
        elif c in ("text2","tt2","dong2","line2","title2"): m[c]="text2"
    df=df.rename(columns=m)
    for col in ("id","text1","text2"):
        if col not in df.columns: df[col]=""
    df=df[["id","text1","text2"]].copy()
    for col in df.columns: df[col]=df[col].astype(str).str.strip()
    return df[df["id"]!=""].reset_index(drop=True)

def _match(df,imgs):
    by={}
    for f in imgs:
        try: by[Path(f.name).stem.strip().lower()]=(_compress(f.getvalue()),f.name)
        except: continue
    mapping,names,ok,miss={},{},{},[]
    for pid in df["id"].tolist():
        k=str(pid).strip().lower(); hit=by.get(k)
        if not hit:
            for s,d in by.items():
                if k in s or s in k: hit=d; break
        if hit: mapping[pid]=hit[0];names[pid]=hit[1];ok[pid]=True
        else: miss.append(pid)
    return mapping,names,list(ok.keys()),miss

def _eff_df():
    df=st.session_state.get("df")
    if df is None: return pd.DataFrame()
    df=df.copy(); ov=st.session_state.get("overrides",{})
    for pid,o in ov.items():
        mask=df["id"]==pid
        if o.get("t1") is not None: df.loc[mask,"text1"]=o["t1"]
        if o.get("t2") is not None: df.loc[mask,"text2"]=o["t2"]
    return df

def _cfg(pid=None):
    g={k:st.session_state.get(f"cfg_{k}",v) for k,v in [
        ("top_margin",155),("bottom_margin",55),("side_padding",40),
        ("product_scale",1.0),("center_mode","centroid"),("font_size",27.5)]}
    ov=st.session_state.get("overrides",{}).get(pid or "",{})
    for k in g:
        if ov.get(k) is not None: g[k]=ov[k]
    return ThumbnailConfig(
        top_margin=g["top_margin"],bottom_margin=g["bottom_margin"],side_padding=g["side_padding"],
        font_size=g["font_size"],font_weight=FONT_WEIGHT,text_color=(0,0,0),text_padding=TEXT_PADDING,
        remove_bg_mode="white" if st.session_state.get("remove_bg") else "none",
        white_tolerance=WHITE_TOL,show_background=True,
        center_mode=g["center_mode"],product_scale=g["product_scale"],
        pill_left=PILL_LEFT,pill_height=PILL_HEIGHT,pill1_top=PILL1_TOP,pill2_gap=PILL2_GAP,
        max_pill_right=MAX_PILL_RIGHT,
        shadow_offset_x=SHADOW_X,shadow_offset_y=SHADOW_Y,shadow_blur=SHADOW_BLUR,shadow_opacity=SHADOW_OP,
        font_family=st.session_state.get("cfg_font_family",DEFAULT_FONT_FAMILY),
        tracking=DEFAULT_TRACKING,
        text_y_nudge=TEXT_Y_NUDGE, # TRUYỀN VÀO IMAGE PROCESSOR
        product_shadow=st.session_state.get("product_shadow",True),
        product_shadow_opacity=28,
        product_shadow_blur=12,
        bottom_gradient=st.session_state.get("bottom_gradient",False),
        bottom_gradient_strength=0.07,
    )

def _validate():
    issues=[]; df=_eff_df(); mapping=st.session_state.get("mapping",{})
    if df.empty: return issues
    dupes=df[df["id"].duplicated(keep=False)]["id"].unique().tolist()
    if dupes: issues.append({"severity":"error","type":"Trùng ID","pid":"","message":f"{len(dupes)} ID trùng: {', '.join(dupes[:5])}"})
    for _,row in df.iterrows():
        pid=row["id"]; t1,t2=row["text1"],row["text2"]
        if pid not in mapping: issues.append({"severity":"error","type":"Thiếu ảnh","pid":pid,"message":"Chưa có ảnh"}); continue
        if len((t1+" "+t2).strip())>50: issues.append({"severity":"warn","type":"Text dài","pid":pid,"message":f"{len(t1)+len(t2)} ký tự"})
        if not t1 and not t2: issues.append({"severity":"warn","type":"Trống text","pid":pid,"message":"Không có text"})
        try:
            img=Image.open(io.BytesIO(mapping[pid])); img.verify()
            img=Image.open(io.BytesIO(mapping[pid])); w,h=img.size
            if min(w,h)<MIN_SRC_DIM: issues.append({"severity":"warn","type":"Ảnh nhỏ","pid":pid,"message":f"{w}×{h}"})
        except: issues.append({"severity":"error","type":"Ảnh hỏng","pid":pid,"message":"Không mở được"})
    return issues

def _export_cfg():
    return {"version":APP_VERSION,"exported":time.strftime("%Y-%m-%d %H:%M:%S"),
            "preset":st.session_state.get("preset_name"),
            "global":{k[4:]:st.session_state[k] for k in st.session_state if k.startswith("cfg_")},
            "overrides":st.session_state.get("overrides",{})}

def _empty(icon,msg):
    with st.container(border=True):
        _,c,_=st.columns([1,2,1])
        with c: st.markdown(f"<div style='text-align:center;padding:40px 0;'><div style='font-size:48px;'>{icon}</div><div style='color:#94a3b8;font-size:14px;margin-top:8px;'>{msg}</div></div>",unsafe_allow_html=True)

# ═══ SIDEBAR ═══
with st.sidebar:
    st.markdown("### 🖼️ Thumbnail Builder")
    st.caption(f"v{APP_VERSION} · Tracking Match")
    st.divider()

    st.markdown("**🎨 Preset**")
    sel=st.selectbox("Preset",list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state["preset_name"]),label_visibility="collapsed")
    if sel!=st.session_state["preset_name"]: _apply_preset(sel); st.rerun()

    st.divider()
    st.markdown("**📐 Layout**")
    c1,c2=st.columns(2)
    with c1: st.number_input("Mép trên",0,300,key="cfg_top_margin",step=5)
    with c2: st.number_input("Mép dưới",0,250,key="cfg_bottom_margin",step=5)
    c1,c2=st.columns(2)
    with c1: st.number_input("Padding bên",0,100,key="cfg_side_padding",step=5)
    with c2: st.number_input("Zoom SP",0.5,1.5,key="cfg_product_scale",step=0.05,format="%.2f")
    st.radio("Căn giữa",["centroid","bbox"],key="cfg_center_mode",horizontal=True,
             format_func=lambda x:"Trọng tâm" if x=="centroid" else "Khung")

    st.divider()
    st.markdown("**✒️ Font**")
    try:
        af=list_available_fonts(); fl=list(af.keys())
        di=fl.index("Montserrat-Bold") if "Montserrat-Bold" in fl else 0
        st.selectbox("Loại chữ",fl,index=di,key="cfg_font_family")
    except: st.session_state["cfg_font_family"]=DEFAULT_FONT_FAMILY
    st.number_input("Kích thước",9.0,50.0,key="cfg_font_size",step=0.5,format="%.1f")

    st.divider()
    st.markdown("**🎨 Hiệu ứng**")
    st.toggle("🔲 Bóng sản phẩm",key="product_shadow",value=True)
    st.toggle("🌫️ Gradient đáy",key="bottom_gradient",value=False)
    st.toggle("🧹 Tách nền trắng",key="remove_bg")

    st.divider()
    st.markdown("**📤 Xuất**")
    out_fmt=st.selectbox("Format",["PNG","JPG","WEBP"])
    jpg_q=st.slider("Quality",70,100,92) if out_fmt in ("JPG","WEBP") else 92

    st.divider()
    with st.expander("💾 Lưu/Tải config"):
        st.download_button("📥 Tải config",json.dumps(_export_cfg(),indent=2,ensure_ascii=False),
                          f"config_{time.strftime('%Y%m%d')}.json","application/json",use_container_width=True)
        up=st.file_uploader("📤 Nhập",type=["json"],label_visibility="collapsed",key="cfg_up")
        if up and st.button("Áp dụng",use_container_width=True):
            try:
                d=json.loads(up.getvalue().decode())
                if d.get("preset") in PRESETS: st.session_state["preset_name"]=d["preset"]
                for k,v in (d.get("global") or {}).items(): st.session_state[f"cfg_{k}"]=v
                if "overrides" in d: st.session_state["overrides"]=d["overrides"]
                st.success("OK!"); st.rerun()
            except Exception as e: st.error(str(e))
    st.divider()
    logout_btn()

# ═══ HERO ═══
c1,c2,c3,c4=st.columns([3,1,1,1])
with c1:
    st.markdown("### 🖼️ Thumbnail Builder Pro")
    st.caption("Montserrat Bold 27.5px · Dynamic Tracking · SS4× pill")
df=st.session_state.get("df"); mapping=st.session_state.get("mapping",{})
ov_count=sum(1 for p in mapping if any(k for k in st.session_state.get("overrides",{}).get(p,{}) if k not in ("t1","t2")))
with c2: st.metric("📋 Excel",len(df) if df is not None else 0)
with c3: st.metric("✅ Khớp",len(mapping))
with c4: st.metric("🎯 Custom",ov_count)

# ═══ TABS ═══
tab1,tab2,tab3,tab4,tab5=st.tabs(["📥 Upload","🔬 Test","👁️ Preview","✏️ Chỉnh riêng","📦 Xuất"])

with tab1:
    st.markdown("### 📄 File Excel / CSV")
    c1,c2=st.columns([4,1])
    with c1:
        ef=st.file_uploader("Excel/CSV",type=["xlsx","xls","csv"],label_visibility="collapsed")
    with c2:
        s=pd.DataFrame({"id":["SP001","SP002"],"text1":["SỨC CHỨA 24 CHAI","BLUETOOTH 5.4"],"text2":["GIÁ KỆ GỖ SỒI CAO CẤP","CỔNG SẠC TYPE-C"]})
        b=io.BytesIO()
        with pd.ExcelWriter(b,engine="openpyxl") as w: s.to_excel(w,index=False)
        st.download_button("📥 Mẫu",b.getvalue(),"mau.xlsx",use_container_width=True)
    if ef:
        try: df_new=_read_excel(ef); st.session_state["df"]=df_new; st.success(f"✅ {len(df_new)} SP")
        except Exception as e: st.error(str(e))
    df=st.session_state.get("df")
    if df is not None:
        with st.expander("📝 Sửa text nhanh"):
            ed=st.data_editor(df,use_container_width=True,hide_index=True,num_rows="fixed",key="be",disabled=["id"],height=min(300,45*len(df)+38))
            if not ed.equals(df):
                if st.button("💾 Lưu text",use_container_width=True): st.session_state["df"]=ed.reset_index(drop=True); st.rerun()
    st.divider()
    st.markdown("### 🖼️ Ảnh sản phẩm")
    st.caption("Tên file chứa id. Ảnh > 1600px tự nén.")
    imgs=st.file_uploader("Ảnh",type=["png","jpg","jpeg","webp"],accept_multiple_files=True,label_visibility="collapsed")
    if imgs:
        ok=[f for f in imgs if f.size<=MAX_UPLOAD_MB*1024*1024]; st.session_state["img_files"]=ok
    imf=st.session_state.get("img_files")
    if df is not None and imf:
        mapping,orig_names,matched,unmatched=_match(df,imf)
        st.session_state["mapping"]=mapping; st.session_state["orig_names"]=orig_names
        c1,c2,c3,c4=st.columns(4)
        c1.metric("📋",len(df)); c2.metric("✅",len(matched)); c3.metric("❌",len(unmatched)); c4.metric("🖼️",len(imf))
        if unmatched:
            with st.expander(f"⚠️ {len(unmatched)} thiếu ảnh"): st.write(unmatched)
    if df is not None and st.session_state.get("mapping"):
        st.divider()
        st.markdown("### 🔍 Kiểm tra batch")
        if st.button("🔍 Quét",type="primary"):
            with st.spinner("Đang quét..."):
                iss=_validate()
            if not iss: st.success("🎉 OK!")
            else:
                errs=[i for i in iss if i["severity"]=="error"]; warns=[i for i in iss if i["severity"]=="warn"]
                c1,c2=st.columns(2); c1.metric("❌ Lỗi",len(errs)); c2.metric("⚠️ Cảnh báo",len(warns))
                st.dataframe(pd.DataFrame(iss),use_container_width=True)

with tab2:
    st.markdown("### 🔬 Test nhanh")
    cl,cr=st.columns([1,1],gap="large")
    with cl:
        si=st.file_uploader("Ảnh",type=["png","jpg","jpeg","webp"],key="su")
        sid=st.text_input("ID","DEMO001"); st1=st.text_input("Text 1","SỨC CHỨA 24 CHAI"); st2=st.text_input("Text 2","GIÁ KỆ GỖ SỒI CAO CẤP")
    with cr:
        if si:
            try:
                prod=Image.open(io.BytesIO(si.getvalue())); bg=_bg(); c=_cfg()
                thumb,info=build_thumbnail(prod,st1,st2,bg,c)
                st.image(thumb,use_container_width=True)
                st.caption(f"600×600 · Font {info['font_sizes_used']} · Gốc {prod.size[0]}×{prod.size[1]}")
                fl=out_fmt.lower(); st.download_button(f"⬇️ {sid}.{fl}",pil_to_bytes(thumb,fl,jpg_q),f"{sanitize_filename(sid)}.{fl}",use_container_width=True)
            except Exception as e: st.error(str(e))
        else: _empty("🖼️","Upload ảnh bên trái")

with tab3:
    df=_eff_df(); mapping=st.session_state.get("mapping",{})
    if df.empty or not mapping: _empty("📋","Hoàn tất Upload")
    else:
        c1,c2,c3=st.columns([3,1,1])
        with c1: search=st.text_input("🔍",placeholder="ID hoặc text...",label_visibility="collapsed")
        with c2: cn=st.selectbox("Cột",[2,3,4,6],index=2,label_visibility="collapsed")
        with c3: mn=st.selectbox("Số",[12,24,48,100],index=1,label_visibility="collapsed")
        dfv=df[df["id"].isin(mapping.keys())].copy()
        if search:
            s=search.lower()
            dfv=dfv[dfv["id"].str.lower().str.contains(s,na=False)|dfv["text1"].str.lower().str.contains(s,na=False)|dfv["text2"].str.lower().str.contains(s,na=False)]
        if len(dfv)==0: st.warning("Không tìm thấy.")
        else:
            ova=st.session_state.get("overrides",{}); bg=_bg(); rows=dfv.head(mn).to_dict("records")
            prog=st.progress(0.0); cols=st.columns(cn)
            for i,row in enumerate(rows):
                prog.progress((i+1)/len(rows)); pid=row["id"]
                try:
                    prod=Image.open(io.BytesIO(mapping[pid])); c=_cfg(pid)
                    thumb,info=build_thumbnail(prod,row["text1"],row["text2"],bg,c)
                    with cols[i%cn]:
                        with st.container(border=True):
                            st.image(thumb,use_container_width=True)
                            hc=any(k for k in ova.get(pid,{}) if k not in ("t1","t2"))
                            bd=(" 🎯" if hc else "")+(" ⚡" if info["any_shrunk"] else "")
                            st.markdown(f"**{pid}**{bd}"); m=row["text1"]
                            if row["text2"]: m+=f" · {row['text2']}"
                            st.caption(m[:60])
                except Exception as e:
                    with cols[i%cn]: st.error(f"{pid}: {str(e)[:40]}")
            prog.empty()

with tab4:
    df=_eff_df(); mapping=st.session_state.get("mapping",{})
    if df.empty or not mapping: _empty("✏️","Hoàn tất Upload")
    else:
        st.markdown("### ✏️ Chỉnh riêng từng SP")
        vids=[p for p in df["id"].tolist() if p in mapping]
        c1,c2,c3=st.columns([3,1,1])
        ova=st.session_state.get("overrides",{})
        with c1: pid=st.selectbox("SP",vids,format_func=lambda x:f"🎯 {x}" if any(k for k in ova.get(x,{}) if k not in ("t1","t2")) else x)
        idx=vids.index(pid) if pid in vids else 0
        with c2:
            if st.button("◀",use_container_width=True,disabled=idx==0): st.session_state["_np"]=vids[idx-1]; st.rerun()
        with c3:
            if st.button("▶",use_container_width=True,disabled=idx>=len(vids)-1): st.session_state["_np"]=vids[idx+1]; st.rerun()
        if "_np" in st.session_state: pid=st.session_state.pop("_np")
        row=df[df["id"]==pid].iloc[0]; ov=ova.get(pid,{})
        cl,cr=st.columns([1.1,1],gap="large")
        with cl:
            try:
                prod=Image.open(io.BytesIO(mapping[pid])); c=_cfg(pid)
                thumb,info=build_thumbnail(prod,row["text1"],row["text2"],_bg(),c)
                st.image(thumb,use_container_width=True)
                st.caption(f"600×600 · Font {info['font_sizes_used']} · Gốc {prod.size[0]}×{prod.size[1]}")
            except Exception as e: st.error(str(e))
        with cr:
            with st.container(border=True):
                st.markdown(f"### ⚙️ `{pid}`")
                use_ov=st.toggle("Cấu hình riêng",value=bool(ov) and any(k for k in ov if k not in ("t1","t2")),key=f"uo_{pid}")
                if use_ov:
                    ct=ov.get("top_margin",st.session_state.get("cfg_top_margin",155))
                    cb=ov.get("bottom_margin",st.session_state.get("cfg_bottom_margin",55))
                    cs=ov.get("side_padding",st.session_state.get("cfg_side_padding",40))
                    csc=ov.get("product_scale",st.session_state.get("cfg_product_scale",1.0))
                    cc=ov.get("center_mode",st.session_state.get("cfg_center_mode","centroid"))
                    cf=ov.get("font_size",st.session_state.get("cfg_font_size",27.5))
                    c1,c2=st.columns(2)
                    with c1: nt=st.number_input("Trên",0,300,int(ct),5,key=f"ot_{pid}"); ns=st.number_input("Side",0,100,int(cs),5,key=f"os_{pid}"); nsc=st.number_input("Zoom",0.5,1.5,float(csc),0.05,format="%.2f",key=f"oz_{pid}")
                    with c2: nb=st.number_input("Dưới",0,250,int(cb),5,key=f"ob_{pid}"); nf=st.number_input("Font",9.0,50.0,float(cf),0.5,format="%.1f",key=f"of_{pid}"); nc=st.radio("Căn",["centroid","bbox"],index=0 if cc=="centroid" else 1,horizontal=True,key=f"oc_{pid}",format_func=lambda x:"TT" if x=="centroid" else "BBox")
                    c1,c2=st.columns(2)
                    with c1:
                        if st.button("💾 Lưu",type="primary",use_container_width=True):
                            st.session_state.setdefault("overrides",{}).setdefault(pid,{}).update({"top_margin":nt,"bottom_margin":nb,"side_padding":ns,"product_scale":nsc,"center_mode":nc,"font_size":nf})
                            st.success("OK!"); st.rerun()
                    with c2:
                        if st.button("📋 Copy→",use_container_width=True): st.session_state["_cf"]=pid
                else:
                    if ov and any(k for k in ov if k not in ("t1","t2")):
                        if st.button("🗑️ Xoá config riêng",use_container_width=True):
                            nv={k:v for k,v in ov.items() if k in ("t1","t2")}
                            if nv: st.session_state["overrides"][pid]=nv
                            else: st.session_state["overrides"].pop(pid,None)
                            st.rerun()
                    else: st.info("Bật toggle để tuỳ chỉnh riêng.")
                if st.session_state.get("_cf")==pid:
                    st.divider(); targets=st.multiselect("Chọn SP đích",[x for x in vids if x!=pid])
                    c1,c2=st.columns(2)
                    with c1:
                        if st.button("✅ Áp dụng",type="primary",use_container_width=True,disabled=not targets):
                            sc={k:v for k,v in ov.items() if k not in ("t1","t2")}
                            for t in targets: st.session_state.setdefault("overrides",{}).setdefault(t,{}).update(sc)
                            st.session_state.pop("_cf",None); st.success(f"Áp {len(targets)} SP"); st.rerun()
                    with c2:
                        if st.button("❌ Huỷ",use_container_width=True): st.session_state.pop("_cf",None); st.rerun()
                st.divider()
                nt1=st.text_input("Text 1",row["text1"],key=f"t1_{pid}"); nt2=st.text_input("Text 2",row["text2"],key=f"t2_{pid}")
                if nt1!=row["text1"] or nt2!=row["text2"]:
                    if st.button("💾 Lưu text",use_container_width=True,key=f"st_{pid}"):
                        o=st.session_state.setdefault("overrides",{}).setdefault(pid,{}); o["t1"]=nt1; o["t2"]=nt2
                        bdf=st.session_state["df"]; bdf.loc[bdf["id"]==pid,"text1"]=nt1; bdf.loc[bdf["id"]==pid,"text2"]=nt2
                        st.success("OK!"); st.rerun()

with tab5:
    df=_eff_df(); mapping=st.session_state.get("mapping",{})
    if df.empty or not mapping: _empty("📦","Hoàn tất Upload")
    else:
        nt=len(mapping); no=sum(1 for p in mapping if any(k for k in st.session_state.get("overrides",{}).get(p,{}) if k not in ("t1","t2")))
        c1,c2,c3,c4=st.columns(4)
        c1.metric("🖼️",nt); c2.metric("📐",out_fmt); c3.metric("📏","600²"); c4.metric("🎯",no)
        with st.container(border=True):
            c1,c2=st.columns(2)
            with c1: inc_csv=st.checkbox("📄 CMS (csv+xlsx)",True); inc_src=st.checkbox("🗂️ Ảnh gốc",False)
            with c2: multi=st.multiselect("📏 Size thêm",[s for s in SUPPORTED_SIZES if s!=600],default=[])
            zn=st.text_input("Tên ZIP",f"thumbnails_{time.strftime('%Y%m%d_%H%M')}.zip")
        if st.button("🚀 Tạo ZIP",type="primary",use_container_width=True):
            sizes=[600]+sorted(multi); bg=_bg(); fl=out_fmt.lower(); total=len(mapping)*len(sizes)
            prog=st.progress(0.0); zbuf=io.BytesIO(); cms=[]; errs=[]; t0=time.time(); done=0
            dfex=df[df["id"].isin(mapping.keys())].reset_index(drop=True)
            with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
                for _,row in dfex.iterrows():
                    pid=row["id"]
                    try:
                        prod=Image.open(io.BytesIO(mapping[pid])); c=_cfg(pid)
                        th600,info=build_thumbnail(prod,row["text1"],row["text2"],bg,c)
                        sid=sanitize_filename(pid)
                        for sz in sizes:
                            done+=1; prog.progress(done/total,f"[{done}/{total}] {pid}@{sz}")
                            img=th600 if sz==600 else th600.resize((sz,sz),Image.LANCZOS)
                            sf="thumbnails" if sz==600 else f"thumbnails_{sz}"
                            zf.writestr(f"{sf}/{sid}.{fl}",pil_to_bytes(img,fl,jpg_q))
                        if inc_src:
                            on=st.session_state.get("orig_names",{}); ext=Path(on.get(pid,f"{pid}.png")).suffix or ".png"
                            zf.writestr(f"source/{sid}{ext}",mapping[pid])
                        ho=any(k for k in st.session_state.get("overrides",{}).get(pid,{}) if k not in ("t1","t2"))
                        cms.append({"id":pid,"text1":row["text1"],"text2":row["text2"],"filename":f"{sid}.{fl}",
                                    "sizes":";".join(str(s) for s in sizes),"fonts":";".join(str(s) for s in info["font_sizes_used"]),
                                    "shrunk":"yes" if info["any_shrunk"] else "no","custom":"yes" if ho else "no"})
                    except Exception as e:
                        done+=len(sizes); errs.append(f"{pid}: {str(e)[:80]}")
                        cms.append({"id":pid,"text1":row.get("text1",""),"text2":row.get("text2",""),"filename":"ERROR","sizes":"","fonts":"","shrunk":str(e)[:50],"custom":"no"})
                if inc_csv and cms:
                    cdf=pd.DataFrame(cms); zf.writestr("cms.csv",cdf.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"))
                    xb=io.BytesIO()
                    with pd.ExcelWriter(xb,engine="openpyxl") as w: cdf.to_excel(w,index=False,sheet_name="CMS")
                    zf.writestr("cms.xlsx",xb.getvalue())
                zf.writestr("config.json",json.dumps(_export_cfg(),indent=2,ensure_ascii=False))
            prog.empty()
            if errs: st.warning(f"⚠️ {len(errs)} lỗi")
            else: st.success(f"✅ {len(dfex)} SP × {len(sizes)} size — {time.time()-t0:.1f}s")
            st.download_button("⬇️ TẢI ZIP",zbuf.getvalue(),zn,"application/zip",use_container_width=True)
