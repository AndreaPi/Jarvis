const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

const isValidDate = (value) => /^\d{4}-\d{2}-\d{2}$/.test(value);

const parseReading = (value) => {
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return parsed;
};

const buildClient = () => {
  if (!supabaseUrl || !supabaseServiceKey) {
    throw new Error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
  }
  return createClient(supabaseUrl, supabaseServiceKey, {
    auth: {
      persistSession: false
    }
  });
};

const readBody = (req) => {
  if (!req.body) {
    return {};
  }
  if (typeof req.body === 'string') {
    return JSON.parse(req.body);
  }
  return req.body;
};

module.exports = async (req, res) => {
  res.setHeader('Content-Type', 'application/json');

  if (req.method !== 'GET' && req.method !== 'POST') {
    res.statusCode = 405;
    res.end(JSON.stringify({ error: 'Method not allowed' }));
    return;
  }

  let supabase;
  try {
    supabase = buildClient();
  } catch (error) {
    res.statusCode = 500;
    res.end(JSON.stringify({ error: error.message }));
    return;
  }

  try {
    if (req.method === 'GET') {
      const meterCode = (req.query.meterCode || '').trim();
      if (!meterCode) {
        res.statusCode = 400;
        res.end(JSON.stringify({ error: 'meterCode is required' }));
        return;
      }

      const { data, error } = await supabase
        .from('meter_readings')
        .select('meter_code, reading, reading_date, updated_at')
        .eq('meter_code', meterCode)
        .maybeSingle();

      if (error) {
        throw error;
      }

      res.statusCode = 200;
      res.end(JSON.stringify({ data: data || null }));
      return;
    }

    const body = readBody(req);
    const meterCode = (body.meterCode || '').trim();
    const readingDate = body.readingDate;
    const reading = parseReading(body.reading);

    if (!meterCode) {
      res.statusCode = 400;
      res.end(JSON.stringify({ error: 'meterCode is required' }));
      return;
    }
    if (!isValidDate(readingDate)) {
      res.statusCode = 400;
      res.end(JSON.stringify({ error: 'readingDate must be YYYY-MM-DD' }));
      return;
    }
    if (reading === null || reading < 0) {
      res.statusCode = 400;
      res.end(JSON.stringify({ error: 'reading must be a positive integer' }));
      return;
    }

    const { data: existing, error: existingError } = await supabase
      .from('meter_readings')
      .select('reading, reading_date')
      .eq('meter_code', meterCode)
      .maybeSingle();

    if (existingError) {
      throw existingError;
    }

    if (existing) {
      if (readingDate < existing.reading_date) {
        res.statusCode = 409;
        res.end(
          JSON.stringify({
            error: 'stale_date',
            message: 'Reading date is earlier than the last saved reading.'
          })
        );
        return;
      }
      if (readingDate >= existing.reading_date && reading < existing.reading) {
        res.statusCode = 409;
        res.end(
          JSON.stringify({
            error: 'reading_decrease',
            message: 'Reading decreased compared to the last saved reading.'
          })
        );
        return;
      }
    }

    const { data: saved, error: saveError } = await supabase
      .from('meter_readings')
      .upsert(
        {
          meter_code: meterCode,
          reading,
          reading_date: readingDate
        },
        { onConflict: 'meter_code' }
      )
      .select('meter_code, reading, reading_date, updated_at')
      .single();

    if (saveError) {
      throw saveError;
    }

    res.statusCode = 200;
    res.end(JSON.stringify({ data: saved }));
  } catch (error) {
    res.statusCode = 500;
    res.end(JSON.stringify({ error: error.message || 'Unexpected error' }));
  }
};
