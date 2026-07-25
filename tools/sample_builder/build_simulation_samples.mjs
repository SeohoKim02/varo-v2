import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = path.resolve(import.meta.dirname, "..", "..");
const OUTPUT_DIR = path.join(ROOT, "samples");

const STORE_POOL = [
  ["S01", "연신내점", "은평구", 37.6190, 126.9211],
  ["S02", "불광점", "은평구", 37.6108, 126.9290],
  ["S03", "응암점", "은평구", 37.5984, 126.9156],
  ["S04", "홍제점", "서대문구", 37.5891, 126.9430],
  ["S05", "신촌점", "서대문구", 37.5563, 126.9368],
  ["S06", "마포점", "마포구", 37.5395, 126.9457],
  ["S07", "종로점", "종로구", 37.5728, 126.9793],
  ["S08", "용산점", "용산구", 37.5326, 126.9652],
  ["S09", "상암점", "마포구", 37.5782, 126.8924],
  ["S10", "구파발점", "은평구", 37.6368, 126.9187],
];

const DC_POOL = [
  ["DC01", "서울 서북권 물류센터", "서대문구", 37.5758, 126.9362],
  ["DC02", "서울 도심권 물류센터", "종로구", 37.5688, 126.9821],
];

const PRODUCTS = [
  ["P001", "냉동만두500g", "냉동식품", "냉동", 4200, 900, 180],
  ["P002", "유어스샐러드랩", "신선식품", "냉장", 5200, 1100, 5],
  ["P003", "브레디크소금버터롤", "베이커리", "상온", 2800, 600, 4],
  ["P004", "서울우유200ml", "유제품", "냉장", 1600, 400, 10],
  ["P005", "혜자로운도시락", "간편식", "냉장", 5900, 1500, 3],
  ["P006", "냉동삼각김밥", "냉동식품", "냉동", 2400, 500, 120],
  ["P007", "제주삼다수2L", "음료", "상온", 2200, 250, 365],
  ["P008", "오뚜기진라면컵", "가공식품", "상온", 1800, 300, 240],
];

const RECOMMENDATION_BLUEPRINTS = {
  edge_3stores_1dc: [
    [0, 1, "DIRECT", null], [1, 2, "VIA_DC", 0], [2, 0, "DIRECT", null],
  ],
  small_4stores_1dc: [
    [0, 2, "DIRECT", null], [1, 3, "DIRECT", null], [3, 0, "DIRECT", null],
    [2, 1, "VIA_DC", 0], [0, 3, "DIRECT", null],
  ],
  normal_6stores_1dc: [
    [0, 3, "DIRECT", null], [1, 4, "VIA_DC", 0], [5, 2, "DIRECT", null],
    [3, 0, "VIA_DC", 0], [2, 4, "DIRECT", null], [4, 5, "VIA_DC", 0],
  ],
  standard_8stores_1dc: [
    [0, 3, "DIRECT", null], [1, 6, "VIA_DC", 0], [7, 2, "DIRECT", null],
    [3, 5, "VIA_DC", 0], [4, 0, "DIRECT", null], [6, 4, "VIA_DC", 0],
    [2, 7, "DIRECT", null], [5, 1, "DIRECT", null],
  ],
  dual_dc_10stores_2dc: [
    [0, 4, "VIA_DC", 0], [6, 9, "VIA_DC", 1], [8, 2, "DIRECT", null],
    [3, 7, "VIA_DC", 1], [5, 1, "DIRECT", null], [9, 6, "VIA_DC", 0],
    [2, 8, "DIRECT", null], [4, 0, "VIA_DC", 1], [1, 5, "DIRECT", null],
    [7, 3, "VIA_DC", 0],
  ],
};

const SPECS = [
  ["small_4stores_1dc", "Varo_V2_sample_small_4stores_1dc.xlsx", 4, 1],
  ["normal_6stores_1dc", "Varo_V2_sample_normal_6stores_1dc.xlsx", 6, 1],
  ["standard_8stores_1dc", "Varo_V2_sample_standard_8stores_1dc.xlsx", 8, 1],
  ["dual_dc_10stores_2dc", "Varo_V2_sample_dual_dc_10stores_2dc.xlsx", 10, 2],
  ["edge_3stores_1dc", "Varo_V2_sample_edge_3stores_1dc.xlsx", 3, 1],
];

const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

function nodeRecord([id, name, region, latitude, longitude], type) {
  return {
    store_id: id,
    store_name: name,
    type,
    store_type: type,
    node_type: type,
    region,
    latitude,
    longitude,
    available_start: "08:00",
    available_end: "22:00",
    capacity: type === "DC" ? 5000 : 500,
    cold_storage_available: true,
    node_id: id,
    node_name: name,
  };
}

function productRecords() {
  return PRODUCTS.map(([product_id, product_name, category, storage_type, unit_price, disposal_cost_per_unit, shelf_life_days]) => ({
    product_id,
    product_name,
    category,
    inventory_category: category,
    storage_type,
    unit_cost: Math.round(unit_price * 0.58),
    unit_price,
    disposal_cost_per_unit,
    disposal_cost: disposal_cost_per_unit,
    shelf_life_days,
    distance_cutline_km: storage_type === "냉장" ? 18 : storage_type === "냉동" ? 25 : 35,
    cold_required: storage_type !== "상온",
    lead_time_days: storage_type === "상온" ? 2 : 1,
    order_cost: 25000,
    holding_rate: 0.18,
  }));
}

function inventoryRecords(stores) {
  const rows = [];
  stores.forEach((store, storeIndex) => {
    PRODUCTS.forEach((product, productIndex) => {
      const [productId, productName, category, storageType, unitPrice, disposalCost] = product;
      const stock = 42 + ((storeIndex * 19 + productIndex * 13) % 84);
      const sales7d = 8 + ((storeIndex * 5 + productIndex * 7) % 28);
      const sales30d = sales7d * 4 + ((storeIndex + productIndex) % 9);
      const expiryDays = 3 + ((storeIndex * 3 + productIndex * 5) % 28);
      rows.push({
        inventory_id: `INV-${store.node_id}-${productId}`,
        store_id: store.node_id,
        store_name: store.node_name,
        region: store.region,
        product_id: productId,
        product_name: productName,
        category,
        inventory_category: stock > sales30d * 1.5 ? "과잉" : stock < sales7d ? "부족" : "정상",
        stock_qty: stock,
        current_stock: stock,
        quantity: stock,
        inventory_qty: stock,
        avg_inventory: Math.round(stock * 0.82),
        avg_daily_sales: Number((sales30d / 30).toFixed(2)),
        sales_7d: sales7d,
        sales_30d: sales30d,
        sales_30: sales30d,
        dead_stock_qty: Math.max(0, stock - sales30d),
        demand_qty: Math.round(sales7d * 1.15),
        demand_forecast_7d: Math.round(sales7d * 1.15),
        days_to_expiry: expiryDays,
        expiry_days: expiryDays,
        expiry_date: new Date(Date.UTC(2026, 6, 1 + expiryDays)),
        inbound_days_ago: 2 + ((storeIndex + productIndex) % 12),
        inbound_days: 2 + ((storeIndex + productIndex) % 12),
        unit_cost: Math.round(unitPrice * 0.58),
        unit_price: unitPrice,
        disposal_cost_per_unit: disposalCost,
        disposal_cost: disposalCost,
        demand_std: Number((1.2 + ((storeIndex + productIndex) % 6) * 0.35).toFixed(2)),
        lead_time_days: storageType === "상온" ? 2 : 1,
        order_cost: 25000,
        holding_cost: Math.round(unitPrice * 0.18),
        daily_holding_cost: Number((unitPrice * 0.18 / 365).toFixed(2)),
        cold_required: storageType !== "상온",
        service_level: 0.95,
        capacity: 500,
        expiry_risk_score: Math.max(0, Math.round(100 - expiryDays * 3)),
        sales_qty: Number((sales30d / 30).toFixed(2)),
      });
    });
  });
  return rows;
}

function distanceKm(a, b) {
  const lat = (a.latitude - b.latitude) * 111;
  const lon = (a.longitude - b.longitude) * 88;
  return Number(Math.max(0.8, Math.sqrt(lat * lat + lon * lon)).toFixed(1));
}

function pathMetrics(a, b, mode = "일반 탑차") {
  const distance = distanceKm(a, b);
  return {
    distance,
    time: Math.round(7 + distance * (mode.includes("냉동") ? 4.5 : 4.0)),
    cost: Math.round((3400 + distance * (mode.includes("냉동") ? 980 : 820)) / 100) * 100,
  };
}

function routeRecords(stores, dcs) {
  const rows = [];
  let serial = 1;
  for (const source of stores) {
    for (const target of stores) {
      if (source.node_id === target.node_id) continue;
      const metrics = pathMetrics(source, target, "일반 탑차");
      rows.push({
        route_id: `PATH${String(serial++).padStart(4, "0")}`,
        from_id: source.node_id,
        to_id: target.node_id,
        source_id: source.node_id,
        source_name: source.node_name,
        target_id: target.node_id,
        target_name: target.node_name,
        source_store: source.node_name,
        target_store: target.node_name,
        route_type: "DIRECT",
        distance_km: metrics.distance,
        route_distance_km: metrics.distance,
        direct_distance_km: metrics.distance,
        travel_time_min: metrics.time,
        route_time_min: metrics.time,
        time_min: metrics.time,
        transport_cost: metrics.cost,
        estimated_cost: metrics.cost,
        direct_cost: metrics.cost,
        cost_per_km: 820,
        fixed_cost: 3400,
        cold_chain_available: true,
        available: true,
        available_start: "08:00",
        available_end: "22:00",
        transport_mode: "일반 탑차",
        transport_type: "일반 탑차",
      });
    }
  }
  for (const dc of dcs) {
    for (const store of stores) {
      for (const [source, target] of [[store, dc], [dc, store]]) {
        const metrics = pathMetrics(source, target, "냉동/냉장 탑차");
        rows.push({
          route_id: `PATH${String(serial++).padStart(4, "0")}`,
          from_id: source.node_id,
          to_id: target.node_id,
          source_id: source.node_id,
          source_name: source.node_name,
          target_id: target.node_id,
          target_name: target.node_name,
          source_store: source.node_name,
          target_store: target.node_name,
          route_type: "VIA_DC",
          distance_km: metrics.distance,
          route_distance_km: metrics.distance,
          direct_distance_km: metrics.distance,
          travel_time_min: metrics.time,
          route_time_min: metrics.time,
          time_min: metrics.time,
          transport_cost: metrics.cost,
          estimated_cost: metrics.cost,
          direct_cost: metrics.cost,
          cost_per_km: 980,
          fixed_cost: 3400,
          cold_chain_available: true,
          available: true,
          available_start: "08:00",
          available_end: "22:00",
          transport_mode: "냉동/냉장 탑차",
          transport_type: "냉동/냉장 탑차",
        });
      }
    }
  }
  return rows;
}

function recommendationRecords(key, stores, dcs) {
  return RECOMMENDATION_BLUEPRINTS[key].map(([sourceIndex, targetIndex, routeType, dcIndex], index) => {
    const source = stores[sourceIndex];
    const target = stores[targetIndex];
    const dc = dcIndex === null ? null : dcs[dcIndex];
    const product = PRODUCTS[index % PRODUCTS.length];
    const direct = pathMetrics(source, target, product[3] === "상온" ? "일반 탑차" : "냉동/냉장 탑차");
    const first = dc ? pathMetrics(source, dc, "냉동/냉장 탑차") : null;
    const second = dc ? pathMetrics(dc, target, "냉동/냉장 탑차") : null;
    const distance = dc ? Number((first.distance + second.distance).toFixed(1)) : direct.distance;
    const time = dc ? first.time + second.time + 4 : direct.time;
    const cost = dc ? first.cost + second.cost : direct.cost;
    const qty = 24 + ((index * 17 + stores.length * 3) % 66);
    const grossBenefit = qty * (product[4] + product[5]);
    const saving = Math.max(8500, Math.round((grossBenefit * (0.62 + (index % 3) * 0.06) - cost) / 100) * 100);
    const vhs = Number((91.4 - index * 2.7 - (routeType === "VIA_DC" ? 0.8 : 0)).toFixed(1));
    const confidence = Number((94.0 - index * 1.8).toFixed(1));
    const grade = vhs >= 85 ? "최적" : vhs >= 75 ? "권장" : "검토";
    return {
      route_id: `R${String(index + 1).padStart(3, "0")}`,
      recommendation_rank: index + 1,
      rank: index + 1,
      product_id: product[0],
      product_name: product[1],
      source_id: source.node_id,
      source_name: source.node_name,
      target_id: target.node_id,
      target_name: target.node_name,
      route_type: routeType,
      dc_id: dc ? dc.node_id : null,
      dc_name: dc ? dc.node_name : null,
      recommended_qty: qty,
      transport_type: product[3] === "상온" && !dc ? "일반 탑차" : "냉동/냉장 탑차",
      transport_mode: product[3] === "상온" && !dc ? "일반 탑차" : "냉동/냉장 탑차",
      estimated_cost: cost,
      transport_cost: cost,
      expected_saving: saving,
      distance_km: distance,
      travel_time_min: time,
      time_min: time,
      vhs_score: vhs,
      candidate_score: Number((96.0 - index * 2.2).toFixed(1)),
      recommendation_grade: grade,
      confidence_score: confidence,
      confidence,
      greedy_action: "재고 이동",
      varo_action: "재고 이동",
      dqn_action: "미연결",
      reason: dc
        ? `${source.node_name}의 과잉 재고를 ${dc.node_name} 경유로 ${target.node_name}에 재배치합니다.`
        : `${source.node_name}의 과잉 재고를 ${target.node_name}에 직접 재배치합니다.`,
      avoided_disposal_cost: Math.round(qty * product[5]),
      recovered_margin: Math.round(qty * product[4] * 0.42),
      status: "READY",
    };
  });
}

function transportModes() {
  return [
    { transport_type: "일반 탑차", base_cost: 3400, cost_per_km: 820, capacity: 220, speed_factor: 1.0, cold_chain: false, icon: "truck", max_distance_km: 35 },
    { transport_type: "냉동/냉장 탑차", base_cost: 3400, cost_per_km: 980, capacity: 180, speed_factor: 0.9, cold_chain: true, icon: "cold_truck", max_distance_km: 30 },
  ];
}

function configRows() {
  return [
    { key: "distance_cutline_km", value: 25, description: "기본 이동 거리 컷라인" },
    { key: "available_start", value: "08:00", description: "기본 거래 시작 시간" },
    { key: "available_end", value: "22:00", description: "기본 거래 종료 시간" },
    { key: "simulation_top_routes", value: 3, description: "홈 기본 애니메이션 경로 수" },
  ];
}

function qualityRows(spec, stores, dcs, recommendations) {
  return [
    { check_item: "필수 시트", result: "PASS", detail: "stores, products, inventory, routes, v2_recommendations 포함" },
    { check_item: "점포 수", result: "PASS", detail: `${stores.length}개` },
    { check_item: "DC 수", result: "PASS", detail: `${dcs.length}개` },
    { check_item: "추천 수", result: "PASS", detail: `${recommendations.length}개` },
    { check_item: "DQN 상태", result: "EXCLUDED", detail: "과거 학습 결과 미사용, dqn_action 미연결" },
    { check_item: "시뮬레이션 용도", result: "PASS", detail: spec },
  ];
}

function readmeRows(spec, stores, dcs, recommendations) {
  return [
    { item: "샘플 ID", description: spec },
    { item: "용도", description: "Varo V2 동적 물류 네트워크 시뮬레이션 검수" },
    { item: "구성", description: `점포 ${stores.length}개, DC ${dcs.length}개, 추천 ${recommendations.length}개` },
    { item: "경로", description: "DIRECT와 VIA_DC를 route_type으로 명시" },
    { item: "DQN", description: "미연결 상태이며 추천 점수에 반영하지 않음" },
    { item: "지도", description: "위도/경도는 SVG 배치 검수용이며 외부 지도 API를 사용하지 않음" },
  ];
}

function toMatrix(rows) {
  if (!rows.length) return { headers: [], values: [] };
  const headers = Object.keys(rows[0]);
  return { headers, values: rows.map((row) => headers.map((key) => row[key] ?? null)) };
}

function columnLetter(index) {
  let value = index + 1;
  let letters = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    letters = String.fromCharCode(65 + remainder) + letters;
    value = Math.floor((value - 1) / 26);
  }
  return letters;
}

function writeSheet(workbook, name, rows, tableName) {
  const sheet = workbook.worksheets.add(name);
  const { headers, values } = toMatrix(rows);
  if (!headers.length) return sheet;
  const all = [headers, ...values];
  const end = columnLetter(headers.length - 1);
  sheet.getRange(`A1:${end}${all.length}`).values = all;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const header = sheet.getRange(`A1:${end}1`);
  header.format = {
    fill: "#344054",
    font: { bold: true, color: "#FFFFFF", name: "Malgun Gothic", size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#667085" },
  };
  header.format.rowHeight = 28;
  if (values.length) {
    const body = sheet.getRange(`A2:${end}${all.length}`);
    body.format.font = { name: "Malgun Gothic", size: 9, color: "#344054" };
    body.format.verticalAlignment = "center";
    body.format.borders = {
      insideHorizontal: { style: "thin", color: "#EAECF0" },
      bottom: { style: "thin", color: "#D0D5DD" },
    };
    sheet.tables.add(`A1:${end}${all.length}`, true, tableName).style = "TableStyleMedium2";
  }
  headers.forEach((headerName, index) => {
    const longest = Math.max(
      String(headerName).length,
      ...values.map((row) => String(row[index] ?? "").length),
    );
    const width = headerName === "reason" || name === "README"
      ? 42
      : Math.min(28, Math.max(10, longest + 2));
    sheet.getRange(`${columnLetter(index)}:${columnLetter(index)}`).format.columnWidth = width;
    if (["unit_price", "unit_cost", "disposal_cost_per_unit", "disposal_cost", "estimated_cost", "transport_cost", "expected_saving", "recovered_margin", "avoided_disposal_cost", "base_cost", "cost_per_km", "order_cost"].includes(headerName)) {
      sheet.getRange(`${columnLetter(index)}2:${columnLetter(index)}${all.length}`).format.numberFormat = "#,##0";
    }
    if (["distance_km", "latitude", "longitude", "vhs_score", "candidate_score", "confidence_score", "confidence"].includes(headerName)) {
      sheet.getRange(`${columnLetter(index)}2:${columnLetter(index)}${all.length}`).format.numberFormat = "0.0";
    }
    if (["latitude", "longitude"].includes(headerName)) {
      sheet.getRangeByIndexes(1, index, values.length, 1).format.numberFormat = "0.0000";
    }
    if (headerName === "expiry_date") {
      sheet.getRange(`${columnLetter(index)}2:${columnLetter(index)}${all.length}`).format.numberFormat = "yyyy-mm-dd";
    }
  });
  return sheet;
}

function hasFormulaError(value) {
  return typeof value === "string" && /#(REF!|DIV\/0!|VALUE!|NAME\?|N\/A)/.test(value);
}

async function buildSample([key, filename, storeCount, dcCount]) {
  const stores = STORE_POOL.slice(0, storeCount).map((row) => nodeRecord(row, "STORE"));
  const dcs = DC_POOL.slice(0, dcCount).map((row) => nodeRecord(row, "DC"));
  const products = productRecords();
  const inventory = inventoryRecords(stores);
  const routes = routeRecords(stores, dcs);
  const recommendations = recommendationRecords(key, stores, dcs);

  const workbook = Workbook.create();
  const sheets = [
    ["stores", [...dcs, ...stores], `Stores_${key}`],
    ["products", products, `Products_${key}`],
    ["inventory", inventory, `Inventory_${key}`],
    ["routes", routes, `Routes_${key}`],
    ["v2_recommendations", recommendations, `Recommendations_${key}`],
    ["transport_modes", transportModes(), `Transport_${key}`],
    ["config", configRows(), `Config_${key}`],
    ["Quality_Check", qualityRows(key, stores, dcs, recommendations), `Quality_${key}`],
    ["README", readmeRows(key, stores, dcs, recommendations), `Readme_${key}`],
  ];

  for (const [sheetName, rows, tableName] of sheets) {
    if (rows.some((row) => Object.values(row).some(hasFormulaError))) {
      throw new Error(`${key}/${sheetName} contains a formula error token`);
    }
    writeSheet(workbook, sheetName, rows, tableName);
  }

  const inspection = await workbook.inspect({
    kind: "sheet,table",
    maxChars: 4000,
    tableMaxRows: 3,
    tableMaxCols: 6,
  });
  const renderResults = [];
  for (const [sheetName] of sheets) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 0.7, format: "png" });
    const bytes = new Uint8Array(await preview.arrayBuffer());
    if (bytes.byteLength < 500) throw new Error(`${key}/${sheetName} render is unexpectedly small`);
    renderResults.push({ sheet: sheetName, bytes: bytes.byteLength });
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  const target = path.join(OUTPUT_DIR, filename);
  await output.save(target);
  return {
    key,
    filename,
    storeCount,
    dcCount,
    productCount: products.length,
    inventoryCount: inventory.length,
    routeCount: routes.length,
    recommendationCount: recommendations.length,
    viaDcs: [...new Set(recommendations.filter((row) => row.route_type === "VIA_DC").map((row) => row.dc_id))],
    renderedSheets: renderResults.length,
    inspectionChars: String(inspection?.ndjson ?? inspection ?? "").length,
    output: target,
  };
}

await fs.mkdir(OUTPUT_DIR, { recursive: true });
const results = [];
for (const spec of SPECS) results.push(await buildSample(spec));
console.log(JSON.stringify(results, null, 2));
