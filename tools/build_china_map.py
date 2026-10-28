# -*- coding: utf-8 -*-
"""从 corpus_manifest 重新生成中国政策分布地图。

用法: python tools/build_china_map.py
输入: data/national/corpus_manifest.json + output/china_map/region_map.json + template.html
输出: output/china_map/region_counts.json + 低空政策采集分布地图.html
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP_DIR = ROOT / "output/china_map"
MANIFEST = ROOT / "data/national/corpus_manifest.json"
REGION_MAP = MAP_DIR / "region_map.json"
COUNTS = MAP_DIR / "region_counts.json"
TEMPLATE = MAP_DIR / "template.html"
OUT_HTML = MAP_DIR / "低空政策采集分布地图.html"

# region_map.json 未收录的城市兜底映射(手动维护)
EXTRA_CITY2PROV = {
    "宜兴市": "江苏省", "大理白族自治州": "云南省", "温州市": "浙江省",
    "济南市": "山东省", "青岛市": "山东省", "株洲市": "湖南省",
    "惠州市": "广东省", "中山市": "广东省", "东莞市": "广东省",
    "佛山市": "广东省", "珠海市": "广东省", "湛江市": "广东省",
    "南通市": "广东省", "南京市": "江苏省", "苏州市": "江苏省",
    "无锡市": "江苏省", "常州市": "江苏省", "扬州市": "江苏省",
    "徐州市": "江苏省", "盐城市": "江苏省", "泰州市": "江苏省",
    "杭州市": "浙江省", "宁波市": "浙江省", "金华市": "浙江省",
    "绍兴市": "浙江省", "嘉兴市": "浙江省", "合肥市": "安徽省",
    "芜湖市": "安徽省", "福州市": "福建省", "厦门市": "福建省",
    "南昌市": "江西省", "济南市章丘区": "山东省", "郑州市": "河南省",
    "武汉市": "湖北省", "十堰市": "湖北省", "长沙市": "湖南省",
    "张家界市": "湖南省", "广州市": "广东省", "深圳市": "广东省",
    "成都市": "四川省", "重庆市": "重庆市", "昆明市": "云南省",
    "贵阳市": "贵州省", "太原市": "山西省", "长治市": "山西省",
    "沈阳市": "辽宁省", "大连市": "辽宁省", "朝阳市": "辽宁省",
    "长春市": "吉林省", "哈尔滨市": "黑龙江省", "西安市": "陕西省",
    "榆林市": "陕西省", "兰州市": "甘肃省", "西宁市": "青海省",
    "银川市": "宁夏回族自治区", "乌鲁木齐市": "新疆维吾尔自治区",
    "呼和浩特市": "内蒙古自治区", "南宁市": "广西壮族自治区",
    "海口市": "海南省", "三亚市": "海南省", "廊坊市": "河北省",
    "青岛市城阳区": "山东省", "上海市金山区": "上海市", "上海市青浦区": "上海市",
    "安阳市": "河南省", "安阳市北关区": "河南省", "宜宾市": "四川省",
    "犍为县": "四川省", "都江堰市": "四川省", "蒲江县": "四川省",
    "邛崃市": "四川省", "曲靖市": "云南省", "昭通市": "云南省",
    "利川市": "湖北省", "延边朝鲜族自治州": "吉林省",
    "广州市天河区": "广东省", "广州市黄埔区": "广东省",
    "深圳市光明区": "广东省", "深圳市前海深港现代服务业合作区": "广东省",
    "深圳市大鹏新区": "广东省", "深圳市福田区": "广东省",
    "深圳市龙岗区": "广东省", "深圳市南山区": "广东省",
}


def norm_region(region):
    """region -> (级别, 省名) 级别: national/province/city/unknown"""
    r = str(region).strip()
    if r in ("全国", "国家", ""):
        return "national", None
    return "local", r


def city_to_prov(city, rmap):
    if city in rmap["prov"]:
        return city
    if city in rmap["city"]:
        return rmap["city"][city]
    if city in EXTRA_CITY2PROV:
        return EXTRA_CITY2PROV[city]
    # 兜底: 尝试去掉省前缀后查
    for prov in rmap["prov"]:
        if city.startswith(prov[:2]) and city != prov:
            # "黑龙江省XX市" 等带前缀
            sub = city[len(prov):]
            if sub in rmap["city"]:
                return prov
    # 直辖市区县
    for d in ("北京市", "上海市", "天津市", "重庆市"):
        if city.startswith(d[:2]):
            return d
    return None


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pol = manifest["policies_detail"]
    rmap = json.loads(REGION_MAP.read_text(encoding="utf-8"))

    prov_cnt, city_cnt, unmatched = {}, {}, []
    nat_n, exp_n, local_n = 0, 0, 0
    for p in pol:
        status = p.get("status", "")
        if status not in ("现行有效", "有效", ""):
            exp_n += 1
        level, r = norm_region(p.get("region", ""))
        if level == "national":
            nat_n += 1
            continue
        local_n += 1
        prov = city_to_prov(r, rmap)
        if prov is None:
            unmatched.append(r)
            continue
        prov_cnt[prov] = prov_cnt.get(prov, 0) + 1
        city_cnt[r] = city_cnt.get(r, 0) + 1

    total = len(pol)
    data = {
        "total": total,
        "national": nat_n,
        "province": dict(sorted(prov_cnt.items(), key=lambda x: -x[1])),
        "city": dict(sorted(city_cnt.items(), key=lambda x: (-x[1], x[0]))),
        "cities_count": len(city_cnt),
        "provinces_count": len(prov_cnt),
        "expired_files": exp_n,
    }
    COUNTS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    prov_arr = [[k, v] for k, v in data["province"].items()]
    city_arr = [[k, v] for k, v in data["city"].items()]
    html = TEMPLATE.read_text(encoding="utf-8")
    echarts = (MAP_DIR / "echarts.min.js").read_text(encoding="utf-8")
    geo = (MAP_DIR / "china.json").read_text(encoding="utf-8")
    html = (html
            .replace("__ECHARTS__", echarts)
            .replace("__GEOJSON__", geo)
            .replace("__DATA__", json.dumps({"prov": prov_arr, "city": city_arr}, ensure_ascii=False))
            .replace("__TOTAL__", str(total))
            .replace("__NAT__", str(nat_n))
            .replace("__LOCAL__", str(local_n))
            .replace("__NPROV__", str(len(prov_cnt)))
            .replace("__NCITY__", str(len(city_cnt)))
            .replace("__EXP__", str(exp_n)))
    OUT_HTML.write_text(html, encoding="utf-8")

    print(f"地图已重建: {OUT_HTML.name}")
    print(f"  政策总数 {total} | 全国级 {nat_n} | 地方 {local_n} 篇覆盖 {len(prov_cnt)} 省 {len(city_cnt)} 市/区 | 失效 {exp_n}")
    if unmatched:
        print(f"  ⚠️ {len(unmatched)} 条 region 未匹配省(需补 EXTRA_CITY2PROV):")
        for u in sorted(set(unmatched)):
            print("     -", u)


if __name__ == "__main__":
    main()
